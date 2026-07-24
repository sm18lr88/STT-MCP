"""Lazy ownership and cancellation of the persistent CUDA worker."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, ClassVar, Final, final
from uuid import uuid4

import anyio
from anyio.streams.buffered import BufferedByteReceiveStream

from stt_mcp.backend import Backend
from stt_mcp.cuda_capacity import ensure_cuda_capacity
from stt_mcp.model import MODEL_ID, MODEL_REVISION
from stt_mcp.worker_errors import (
    WORKER_COMMUNICATION_EXCEPTIONS,
    WorkerBusyError,
    WorkerFailureError,
)
from stt_mcp.worker_protocol import (
    MAX_LINE_BYTES,
    ErrorEvent,
    InitializeCommand,
    ModelSpec,
    ProtocolLineTooLargeError,
    ReadyEvent,
    ResultEvent,
    RuntimeSpec,
    ShutdownCommand,
    ShutdownCompleteEvent,
    TranscribeCommand,
    WorkerCommand,
    WorkerProtocolError,
    decode_event,
    encode_message,
)
from stt_mcp.worker_state import OpenWorker, PendingRequest

__all__ = ["WorkerBusyError", "WorkerFailureError", "WorkerSupervisor"]

if TYPE_CHECKING:
    from anyio.abc import ByteReceiveStream, Process

    from stt_mcp.backend import AudioChunk
    from stt_mcp.runtime_policy import RuntimePlan

CREATE_NO_WINDOW: Final = 0x08000000


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

    @property
    def backend(self) -> Backend:
        """Return Granite for transcript provenance."""
        return Backend.GRANITE

    async def start(self) -> None:
        """Lazily initialize the worker before source preprocessing begins."""
        self._acquire()
        try:
            try:
                await self._ensure_started()
            except WORKER_COMMUNICATION_EXCEPTIONS as error:
                await self._invalidate()
                raise WorkerFailureError(
                    detail=f"worker communication failed: {error}"
                ) from error
        finally:
            self._lock.release()

    async def transcribe(self, chunk: AudioChunk) -> str:
        """Transcribe one chunk or reject immediately when already occupied."""
        self._acquire()
        try:
            return await self._transcribe_owned(chunk)
        except anyio.get_cancelled_exc_class():
            await self._invalidate()
            raise
        finally:
            self._lock.release()

    def _acquire(self) -> None:
        try:
            self._lock.acquire_nowait()
        except anyio.WouldBlock:
            raise WorkerBusyError from None

    async def _transcribe_owned(self, chunk: AudioChunk) -> str:
        try:
            await self._ensure_started()
            pending = PendingRequest(request_id=uuid4(), chunk_index=chunk.index)
            await self._send(
                TranscribeCommand(
                    request_id=pending.request_id,
                    chunk_index=pending.chunk_index,
                    audio_path=chunk.path,
                )
            )
            return await self._text_from_event(await self._receive(), pending)
        except WORKER_COMMUNICATION_EXCEPTIONS as error:
            await self._invalidate()
            raise WorkerFailureError(detail=f"worker communication failed: {error}") from error

    async def _text_from_event(
        self,
        event: ReadyEvent | ResultEvent | ErrorEvent | ShutdownCompleteEvent,
        pending: PendingRequest,
    ) -> str:
        if isinstance(event, ResultEvent):
            if event.request_id == pending.request_id and event.chunk_index == pending.chunk_index:
                return event.text
            await self._invalidate()
            raise WorkerFailureError(detail="worker returned an unexpected event")
        if isinstance(event, ErrorEvent):
            if not event.worker_reusable:
                await self._invalidate()
            raise WorkerFailureError(detail=event.message)
        await self._invalidate()
        raise WorkerFailureError(detail="worker returned an unexpected event")

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
                WorkerProtocolError,
            ):
                return
            finally:
                await self._invalidate()

    async def _ensure_started(self) -> None:
        if self._process is not None:
            return
        ensure_cuda_capacity()
        opened = await self._open_process()
        self._process = opened.process
        self._reader = BufferedByteReceiveStream(opened.stdout)
        self._stderr = opened.stderr
        runtime = RuntimeSpec(device="cuda", attention=self._runtime.attention)
        model = ModelSpec(model_id=MODEL_ID, revision=MODEL_REVISION)
        initialized = False
        try:
            await self._send(InitializeCommand(model=model, runtime=runtime))
            await self._confirm_started(await self._receive())
            initialized = True
        finally:
            if not initialized:
                await self._invalidate()

    async def _open_process(self) -> OpenWorker:
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
        return OpenWorker(process=process, stdout=process.stdout, stderr=process.stderr)

    async def _confirm_started(
        self,
        event: ReadyEvent | ResultEvent | ErrorEvent | ShutdownCompleteEvent,
    ) -> None:
        if isinstance(event, ReadyEvent):
            return
        if isinstance(event, ErrorEvent):
            await self._invalidate()
            raise WorkerFailureError(detail=event.message)
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
        except anyio.DelimiterNotFound as error:
            detail = f"worker protocol line exceeded {MAX_LINE_BYTES} bytes"
            raise ProtocolLineTooLargeError(detail=detail) from error
        return decode_event(line)

    async def _closed_worker_detail(self) -> str:
        process = self._process
        stderr = self._stderr
        if process is None:
            return "worker stdout closed unexpectedly"
        try:
            returncode = await process.wait()
            message = await self._stderr_tail(stderr)
            suffix = f": {message}" if message else ""
            return f"worker exited with code {returncode}{suffix}"
        finally:
            await self._invalidate()

    async def _stderr_tail(self, stderr: ByteReceiveStream | None) -> str:
        chunks: list[bytes] = []
        if stderr is not None:
            while True:
                try:
                    chunks.append(await stderr.receive())
                except anyio.EndOfStream:
                    break
        return b"".join(chunks)[-4096:].decode("utf-8", errors="replace").strip()

    def _clear_process(self) -> Process | None:
        process = self._process
        self._process = None
        self._reader = None
        self._stderr = None
        return process

    async def _invalidate(self) -> None:
        process = self._process
        if process is None:
            return
        with anyio.CancelScope(shield=True):
            try:
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                _ = await process.wait()
                await process.aclose()
            finally:
                if self._process is process:
                    _ = self._clear_process()
