"""Persistent CUDA inference worker entry point."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import torch

from stt_mcp.model import MODEL_ID, MODEL_REVISION, GraniteTranscriber, ModelContractError
from stt_mcp.worker_protocol import (
    ErrorEvent,
    InitializeCommand,
    ReadyEvent,
    ResultEvent,
    RuntimeSpec,
    ShutdownCommand,
    ShutdownCompleteEvent,
    TranscribeCommand,
    WorkerErrorCode,
    decode_command,
    encode_message,
)

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID


def main() -> None:
    """Read commands from stdin and emit protocol events on stdout."""
    transcriber: GraniteTranscriber | None = None
    for line in sys.stdin.buffer:
        command = decode_command(line)
        match command:
            case InitializeCommand(model=model, runtime=runtime):
                transcriber = _initialize(model.model_id, model.revision, runtime)
                if transcriber is None:
                    return
                _emit(
                    ReadyEvent(
                        worker_pid=os.getpid(),
                        model=model,
                        runtime=runtime,
                    )
                )
            case TranscribeCommand(
                request_id=request_id,
                chunk_index=chunk_index,
                audio_path=audio_path,
            ):
                if transcriber is None:
                    _emit(
                        ErrorEvent(
                            request_id=request_id,
                            chunk_index=chunk_index,
                            code=WorkerErrorCode.INITIALIZATION_FAILED,
                            message="worker is not initialized",
                            worker_reusable=False,
                        )
                    )
                    return
                if not _transcribe(transcriber, request_id, chunk_index, audio_path):
                    return
            case ShutdownCommand():
                _emit(ShutdownCompleteEvent())
                return


def _initialize(model_id: str, revision: str, runtime: RuntimeSpec) -> GraniteTranscriber | None:
    if model_id != MODEL_ID or revision != MODEL_REVISION:
        _emit(
            ErrorEvent(
                code=WorkerErrorCode.INITIALIZATION_FAILED,
                message="worker model identity is not pinned",
                worker_reusable=False,
            )
        )
        return None
    try:
        return GraniteTranscriber.load(device=runtime.device, attention=runtime.attention)
    except (OSError, RuntimeError, ValueError, ModelContractError) as error:
        _emit(
            ErrorEvent(
                code=WorkerErrorCode.INITIALIZATION_FAILED,
                message=str(error),
                worker_reusable=False,
            )
        )
        return None


def _transcribe(
    transcriber: GraniteTranscriber,
    request_id: UUID,
    chunk_index: int,
    audio_path: Path,
) -> bool:
    try:
        text = transcriber.transcribe(audio_path)
    except torch.OutOfMemoryError as error:
        _emit_error(
            request_id,
            chunk_index,
            WorkerErrorCode.CUDA_OUT_OF_MEMORY,
            error,
            worker_reusable=False,
        )
        return False
    except ModelContractError as error:
        _emit_error(
            request_id,
            chunk_index,
            WorkerErrorCode.INVALID_AUDIO,
            error,
            worker_reusable=True,
        )
        return True
    except RuntimeError as error:
        _emit_error(
            request_id,
            chunk_index,
            WorkerErrorCode.INFERENCE_FAILED,
            error,
            worker_reusable=False,
        )
        return False
    _emit(ResultEvent(request_id=request_id, chunk_index=chunk_index, text=text))
    return True


def _emit_error(
    request_id: UUID,
    chunk_index: int,
    code: WorkerErrorCode,
    error: RuntimeError | ModelContractError,
    *,
    worker_reusable: bool,
) -> None:
    _emit(
        ErrorEvent(
            request_id=request_id,
            chunk_index=chunk_index,
            code=code,
            message=str(error),
            worker_reusable=worker_reusable,
        )
    )


def _emit(event: ReadyEvent | ResultEvent | ErrorEvent | ShutdownCompleteEvent) -> None:
    _ = sys.stdout.buffer.write(encode_message(event))
    _ = sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
