"""Strict versioned NDJSON contract for the CUDA worker."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, override
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from stt_mcp.runtime_policy import AttentionImplementation

MAX_LINE_BYTES: Final = 1_048_576


class _WireModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    protocol: Literal["stt-worker"] = "stt-worker"
    version: Literal[1] = 1


class ModelSpec(BaseModel):
    """Pinned model identity sent across the worker boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    revision: str


class RuntimeSpec(BaseModel):
    """Effective CUDA runtime sent across the worker boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    device: str
    attention: AttentionImplementation


class InitializeCommand(_WireModel):
    """Request lazy model initialization."""

    type: Literal["initialize"] = "initialize"
    model: ModelSpec
    runtime: RuntimeSpec


class TranscribeCommand(_WireModel):
    """Request transcription of one normalized audio chunk."""

    type: Literal["transcribe"] = "transcribe"
    request_id: UUID
    chunk_index: int
    audio_path: Path


class ShutdownCommand(_WireModel):
    """Request graceful worker shutdown."""

    type: Literal["shutdown"] = "shutdown"


class ReadyEvent(_WireModel):
    """Confirm loaded model and effective runtime."""

    type: Literal["ready"] = "ready"
    worker_pid: int
    model: ModelSpec
    runtime: RuntimeSpec


class ResultEvent(_WireModel):
    """Return text for exactly one audio chunk."""

    type: Literal["result"] = "result"
    request_id: UUID
    chunk_index: int
    text: str


class WorkerErrorCode(StrEnum):
    """Stable worker failure categories."""

    INITIALIZATION_FAILED = "initialization_failed"
    INFERENCE_FAILED = "inference_failed"
    INVALID_AUDIO = "invalid_audio"
    CUDA_OUT_OF_MEMORY = "cuda_out_of_memory"


class ErrorEvent(_WireModel):
    """Return a sanitized worker failure."""

    type: Literal["error"] = "error"
    request_id: UUID | None = None
    chunk_index: int | None = None
    code: WorkerErrorCode
    message: str
    worker_reusable: bool


class ShutdownCompleteEvent(_WireModel):
    """Confirm graceful worker shutdown."""

    type: Literal["shutdown_complete"] = "shutdown_complete"


type WorkerCommand = InitializeCommand | TranscribeCommand | ShutdownCommand
type WorkerEvent = ReadyEvent | ResultEvent | ErrorEvent | ShutdownCompleteEvent
type WorkerMessage = WorkerCommand | WorkerEvent

COMMAND_ADAPTER: Final[TypeAdapter[WorkerCommand]] = TypeAdapter(
    Annotated[WorkerCommand, Field(discriminator="type")]
)
EVENT_ADAPTER: Final[TypeAdapter[WorkerEvent]] = TypeAdapter(
    Annotated[WorkerEvent, Field(discriminator="type")]
)


@dataclass(frozen=True, slots=True)
class WorkerProtocolError(Exception):
    """Raised when worker traffic violates the wire contract."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class ProtocolLineTooLargeError(WorkerProtocolError):
    """Raised when one worker message exceeds the bounded line size."""


def encode_message(message: WorkerMessage) -> bytes:
    """Encode one validated message as UTF-8 NDJSON."""
    return message.model_dump_json().encode("utf-8") + b"\n"


def decode_command(line: bytes) -> WorkerCommand:
    """Parse an untrusted command line into a validated command."""
    _enforce_line_size(line)
    try:
        return COMMAND_ADAPTER.validate_json(line)
    except ValidationError as error:
        detail = f"invalid worker command ({error.error_count()} validation errors)"
        raise WorkerProtocolError(detail=detail) from error


def decode_event(line: bytes) -> WorkerEvent:
    """Parse an untrusted event line into a validated event."""
    _enforce_line_size(line)
    try:
        return EVENT_ADAPTER.validate_json(line)
    except ValidationError as error:
        detail = f"invalid worker event ({error.error_count()} validation errors)"
        raise WorkerProtocolError(detail=detail) from error


def _enforce_line_size(line: bytes) -> None:
    if len(line) > MAX_LINE_BYTES:
        detail = f"worker protocol line exceeded {MAX_LINE_BYTES} bytes"
        raise ProtocolLineTooLargeError(detail=detail)
