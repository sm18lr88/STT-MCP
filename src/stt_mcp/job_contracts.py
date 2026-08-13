"""Validated contracts for durable transcription jobs."""

from datetime import datetime
from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from stt_mcp.service import TranscriptionResult


@unique
class TranscriptionPhase(StrEnum):
    """Observable phases of a transcription operation."""

    QUEUED = "queued"
    STARTING_ENGINE = "starting_engine"
    NORMALIZING = "normalizing"
    TRANSCRIBING = "transcribing"
    PUBLISHING = "publishing"


class JobProgress(BaseModel):
    """Latest durable progress snapshot for a transcription job."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    phase: TranscriptionPhase
    completed_chunks: int = Field(ge=0)
    total_chunks: int | None = Field(default=None, ge=1)
    message: str


class JobError(BaseModel):
    """Stable terminal failure exposed to MCP clients."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    code: Literal["transcription_failed"] = "transcription_failed"
    phase: TranscriptionPhase
    message: str


class QueuedJob(BaseModel):
    """Accepted job waiting for its lifespan task to start."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    status: Literal["queued"] = "queued"
    job_id: str
    created_at: datetime
    progress: JobProgress


class RunningJob(BaseModel):
    """Job currently executing one transcription phase."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    status: Literal["running"] = "running"
    job_id: str
    created_at: datetime
    started_at: datetime
    progress: JobProgress


class CancellingJob(BaseModel):
    """Job waiting for inference and process cleanup to finish."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    status: Literal["cancelling"] = "cancelling"
    job_id: str
    created_at: datetime
    started_at: datetime | None
    progress: JobProgress


class CompletedJob(BaseModel):
    """Job with a published transcript result."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    status: Literal["completed"] = "completed"
    job_id: str
    created_at: datetime
    started_at: datetime
    finished_at: datetime
    progress: JobProgress
    result: TranscriptionResult


class FailedJob(BaseModel):
    """Job with a terminal transcription error."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    status: Literal["failed"] = "failed"
    job_id: str
    created_at: datetime
    started_at: datetime
    finished_at: datetime
    progress: JobProgress
    error: JobError


class CancelledJob(BaseModel):
    """Job whose cancellation cleanup has finished."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    status: Literal["cancelled"] = "cancelled"
    job_id: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime
    progress: JobProgress


type TranscriptionJobStatus = Annotated[
    QueuedJob | RunningJob | CancellingJob | CompletedJob | FailedJob | CancelledJob,
    Field(discriminator="status"),
]
