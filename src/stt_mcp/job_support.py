"""Support contracts and errors for durable transcription jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

if TYPE_CHECKING:
    from uuid import UUID

    from stt_mcp.service import (
        ProgressReporter,
        TranscriptionRequest,
        TranscriptionResult,
    )


class JobTranscriber(Protocol):
    """Transcription capability required by the job manager."""

    async def transcribe(
        self,
        request: TranscriptionRequest,
        progress: ProgressReporter | None = None,
    ) -> TranscriptionResult:
        """Run one transcription with optional progress reporting."""
        ...


@dataclass(frozen=True, slots=True)
class JobNotFoundError(Exception):
    """Raised when a retained transcription job cannot be found."""

    job_id: UUID

    @override
    def __str__(self) -> str:
        return f"transcription job not found: {self.job_id}"


@dataclass(frozen=True, slots=True)
class JobConflictError(Exception):
    """Raised when one job ID is reused with different inputs."""

    job_id: UUID

    @override
    def __str__(self) -> str:
        return f"transcription job ID already has different inputs: {self.job_id}"
