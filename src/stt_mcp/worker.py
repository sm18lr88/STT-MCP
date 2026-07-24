"""Persistent CUDA inference worker entry point."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import BinaryIO, ClassVar, final

import torch

from stt_mcp.model import MODEL_ID, MODEL_REVISION, GraniteTranscriber, ModelContractError
from stt_mcp.worker_protocol import (
    ErrorEvent,
    InitializeCommand,
    ReadyEvent,
    ResultEvent,
    ShutdownCommand,
    ShutdownCompleteEvent,
    TranscribeCommand,
    WorkerCommand,
    WorkerErrorCode,
    WorkerProtocolError,
    decode_command,
    encode_message,
)


@dataclass(frozen=True, slots=True)
class _TranscriptionFailure:
    code: WorkerErrorCode
    error: RuntimeError | ModelContractError
    worker_reusable: bool


@final
class _WorkerSession:
    """Mutable protocol state for one worker process."""

    __slots__: ClassVar[tuple[str, ...]] = ("_output_stream", "_transcriber")

    _output_stream: BinaryIO
    _transcriber: GraniteTranscriber | None

    def __init__(self, output_stream: BinaryIO) -> None:
        self._output_stream = output_stream
        self._transcriber = None

    def run(self, input_stream: BinaryIO) -> None:
        for line in input_stream:
            if not self._dispatch(decode_command(line)):
                return

    @singledispatchmethod
    def _dispatch(self, command: WorkerCommand) -> bool:
        raise WorkerProtocolError(detail=f"unsupported command type: {type(command).__name__}")

    @_dispatch.register
    def dispatch_initialize(self, command: InitializeCommand) -> bool:
        return self._initialize(command)

    @_dispatch.register
    def dispatch_transcribe(self, command: TranscribeCommand) -> bool:
        return self._transcribe(command)

    @_dispatch.register
    def dispatch_shutdown(self, _command: ShutdownCommand) -> bool:
        self._emit(ShutdownCompleteEvent())
        return False

    def _initialize(self, command: InitializeCommand) -> bool:
        if command.model.model_id != MODEL_ID or command.model.revision != MODEL_REVISION:
            self._emit(
                ErrorEvent(
                    code=WorkerErrorCode.INITIALIZATION_FAILED,
                    message="worker model identity is not pinned",
                    worker_reusable=False,
                )
            )
            return False
        try:
            self._transcriber = GraniteTranscriber.load(
                device=command.runtime.device,
                attention=command.runtime.attention,
            )
        except (OSError, RuntimeError, ValueError, ModelContractError) as error:
            self._emit(
                ErrorEvent(
                    code=WorkerErrorCode.INITIALIZATION_FAILED,
                    message=str(error),
                    worker_reusable=False,
                )
            )
            return False
        self._emit(
            ReadyEvent(
                worker_pid=os.getpid(),
                model=command.model,
                runtime=command.runtime,
            )
        )
        return True

    def _transcribe(self, command: TranscribeCommand) -> bool:
        transcriber = self._transcriber
        if transcriber is None:
            self._emit(
                ErrorEvent(
                    request_id=command.request_id,
                    chunk_index=command.chunk_index,
                    code=WorkerErrorCode.INITIALIZATION_FAILED,
                    message="worker is not initialized",
                    worker_reusable=False,
                )
            )
            return False
        result = _transcription_result(transcriber, command)
        return self._emit_transcription_result(result, command)

    @singledispatchmethod
    def _emit_transcription_result(
        self,
        result: str | _TranscriptionFailure,
        _command: TranscribeCommand,
    ) -> bool:
        detail = f"unsupported transcription result: {type(result).__name__}"
        raise WorkerProtocolError(detail=detail)

    @_emit_transcription_result.register
    def emit_text(self, text: str, command: TranscribeCommand) -> bool:
        self._emit(
            ResultEvent(
                request_id=command.request_id,
                chunk_index=command.chunk_index,
                text=text,
            )
        )
        return True

    @_emit_transcription_result.register
    def emit_transcription_failure(
        self,
        failure: _TranscriptionFailure,
        command: TranscribeCommand,
    ) -> bool:
        self._emit_failure(command, failure)
        return failure.worker_reusable

    def _emit_failure(
        self,
        command: TranscribeCommand,
        failure: _TranscriptionFailure,
    ) -> None:
        self._emit(
            ErrorEvent(
                request_id=command.request_id,
                chunk_index=command.chunk_index,
                code=failure.code,
                message=str(failure.error),
                worker_reusable=failure.worker_reusable,
            )
        )

    def _emit(self, event: ReadyEvent | ResultEvent | ErrorEvent | ShutdownCompleteEvent) -> None:
        _ = self._output_stream.write(encode_message(event))
        self._output_stream.flush()


def run_worker(input_stream: BinaryIO, output_stream: BinaryIO) -> None:
    """Run one worker protocol session over binary streams."""
    _WorkerSession(output_stream).run(input_stream)


def _transcription_result(
    transcriber: GraniteTranscriber,
    command: TranscribeCommand,
) -> str | _TranscriptionFailure:
    try:
        return transcriber.transcribe(command.audio_path)
    except torch.OutOfMemoryError as error:
        return _TranscriptionFailure(
            code=WorkerErrorCode.CUDA_OUT_OF_MEMORY,
            error=error,
            worker_reusable=False,
        )
    except ModelContractError as error:
        return _TranscriptionFailure(
            code=WorkerErrorCode.INVALID_AUDIO,
            error=error,
            worker_reusable=True,
        )
    except RuntimeError as error:
        return _TranscriptionFailure(
            code=WorkerErrorCode.INFERENCE_FAILED,
            error=error,
            worker_reusable=False,
        )


def main() -> None:
    """Read commands from stdin and emit protocol events on stdout."""
    run_worker(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    main()
