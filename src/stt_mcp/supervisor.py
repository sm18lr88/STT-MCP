"""Lazy ownership and cancellation of the persistent CUDA worker."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, ClassVar, Final, final, override
from uuid import uuid4

import anyio
from anyio.streams.buffered import BufferedByteReceiveStream

from stt_mcp.cuda_capacity import ensure_cuda_capacity
from stt_mcp.model import MODEL_ID, MODEL_REVISION
from stt_mcp.worker_protocol import (
    MAX_LINE_BYTES,
    ErrorEvent,
    InitializeCommand,
    ModelSpec,
    ReadyEvent,
    ResultEvent,
    RuntimeSpec,
    ShutdownCommand,
    ShutdownCompleteEvent,
    TranscribeCommand,
    WorkerCommand,
    decode_event,
    encode_message,
)

if TYPE_CHECKING:
    from anyio.abc import ByteReceiveStream, Process

    from stt_mcp.media import NormalizedChunk
    from stt_mcp.runtime_policy import RuntimePlan

CREATE_NO_WINDOW: Final = 0x08000000


class WorkerBusyError(Exception):
    """Raised when another transcription owns the worker."""

    @override
    def __str__(self) -> str:
        return "STT-MCP is busy with another transcription"


class WorkerFailureError(Exception):
    """Raised when the worker cannot satisfy a request."""

    detail: str

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    @override
    def __str__(self) -> str:
        return self.detail


@final
class WorkerSupervisor:
    """Stateful lazy supervisor for exactly one CUDA worker process."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "_lock",
        "_process",
        "_reader",
        "_runtime",
        "_stderr",
    )

    _lock: anyio.Lock
    _process: Process | None
    _reader: BufferedByteReceiveStream | None
    _runtime: RuntimePlan
    _stderr: ByteReceiveStream | None

    def __init__(self, runtime: RuntimePlan) -> None:
        self._runtime = runtime
        self._lock = anyio.Lock()
        self._process = None
        self._reader = None
        self._stderr = None

    async def start(self) -> None:
        """Lazily initialize the worker before source preprocessing begins."""
        if self._lock.locked():
            raise WorkerBusyError
        async with self._lock:
            await self._ensure_started()

    async def transcribe(self, chunk: NormalizedChunk) -> str:
        """Transcribe one chunk or reject immediately when already occupied."""
        if self._lock.locked():
            raise WorkerBusyError
        async with self._lock:
            try:
                await self._ensure_started()
                request_id = uuid4()
                await self._send(
                    TranscribeCommand(
                        request_id=request_id,
                        chunk_index=chunk.index,
                        audio_path=chunk.path,
                    )
                )
                event = await self._receive()
                match event:
                    case ResultEvent(
                        request_id=event_request_id,
                        chunk_index=chunk_index,
                        text=text,
                    ) if event_request_id == request_id and chunk_index == chunk.index:
                        return text
                    case ErrorEvent(message=message, worker_reusable=worker_reusable):
                        if not worker_reusable:
                            await self._invalidate()
                        raise WorkerFailureError(detail=message)
                    case ReadyEvent() | ShutdownCompleteEvent() | ResultEvent():
                        await self._invalidate()
                        raise WorkerFailureError(detail="worker returned an unexpected event")
            except anyio.get_cancelled_exc_class():
                with anyio.CancelScope(shield=True):
                    await self._invalidate()
                raise

    async def aclose(self) -> None:
        """Gracefully stop the owned worker, then release all process handles."""
        with anyio.CancelScope(shield=True):
            if self._process is None:
                return
            try:
                with anyio.move_on_after(5, shield=True):
                    await self._send(ShutdownCommand())
                    _ = await self._receive()
            except (
                anyio.BrokenResourceError,
                anyio.ClosedResourceError,
                ConnectionResetError,
                WorkerFailureError,
            ):
                pass
            finally:
                await self._invalidate()

    async def _ensure_started(self) -> None:
        if self._process is not None:
            return
        ensure_cuda_capacity()
        environment = dict(os.environ)
        environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        environment["TRANSFORMERS_VERBOSITY"] = "error"
        process = await anyio.open_process(
            (sys.executable, "-m", "stt_mcp.worker"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            _ = await process.wait()
            raise WorkerFailureError(detail="worker stdio pipes were not created")
        self._process = process
        self._reader = BufferedByteReceiveStream(process.stdout)
        self._stderr = process.stderr
        runtime = RuntimeSpec(device="cuda", attention=self._runtime.attention)
        model = ModelSpec(model_id=MODEL_ID, revision=MODEL_REVISION)
        await self._send(InitializeCommand(model=model, runtime=runtime))
        event = await self._receive()
        match event:
            case ReadyEvent():
                return
            case ErrorEvent(message=message):
                await self._invalidate()
                raise WorkerFailureError(detail=message)
            case ResultEvent() | ShutdownCompleteEvent():
                await self._invalidate()
                raise WorkerFailureError(detail="worker did not confirm initialization")

    async def _send(self, command: WorkerCommand) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise WorkerFailureError(detail="worker is not running")
        await process.stdin.send(encode_message(command))

    async def _receive(self) -> ReadyEvent | ResultEvent | ErrorEvent | ShutdownCompleteEvent:
        if self._reader is None:
            raise WorkerFailureError(detail="worker output is unavailable")
        try:
            line = await self._reader.receive_until(b"\n", MAX_LINE_BYTES)
        except anyio.IncompleteRead as error:
            detail = await self._closed_worker_detail()
            raise WorkerFailureError(detail=detail) from error
        return decode_event(line)

    async def _closed_worker_detail(self) -> str:
        process = self._process
        stderr = self._stderr
        if process is None:
            return "worker stdout closed unexpectedly"
        returncode = await process.wait()
        chunks: list[bytes] = []
        if stderr is not None:
            while True:
                try:
                    chunks.append(await stderr.receive())
                except anyio.EndOfStream:
                    break
        await process.aclose()
        self._process = None
        self._reader = None
        self._stderr = None
        message = b"".join(chunks)[-4096:].decode("utf-8", errors="replace").strip()
        suffix = f": {message}" if message else ""
        return f"worker exited with code {returncode}{suffix}"

    async def _invalidate(self) -> None:
        process = self._process
        self._process = None
        self._reader = None
        self._stderr = None
        if process is None:
            return
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        _ = await process.wait()
        await process.aclose()
