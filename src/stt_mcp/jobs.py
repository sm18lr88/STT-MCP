"""Process-lifetime ownership for long-running transcription jobs."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, final

import anyio

from stt_mcp.backend import TranscriptionBusyError
from stt_mcp.job_contracts import (
    CancelledJob,
    CancellingJob,
    CompletedJob,
    FailedJob,
    JobError,
    JobProgress,
    QueuedJob,
    RunningJob,
    TranscriptionJobStatus,
    TranscriptionPhase,
)
from stt_mcp.job_support import JobConflictError, JobNotFoundError, JobTranscriber

if TYPE_CHECKING:
    from uuid import UUID

    from anyio.abc import TaskGroup

    from stt_mcp.service import (
        ProgressReporter,
        TranscriptionRequest,
        TranscriptionResult,
    )

LOGGER = logging.getLogger(__name__)


@final
class _JobRecord:
    __slots__: tuple[str, ...] = ("cancel_scope", "done", "request", "status")

    def __init__(
        self,
        request: TranscriptionRequest,
        status: TranscriptionJobStatus,
        done: anyio.Event,
    ) -> None:
        self.request = request
        self.status = status
        self.done = done
        self.cancel_scope: anyio.CancelScope | None = None


@final
class TranscriptionJobManager:
    """Own transcription jobs for the FastMCP process lifespan."""

    def __init__(self, service: JobTranscriber, task_group: TaskGroup) -> None:
        self._service = service
        self._task_group = task_group
        self._jobs: dict[UUID, _JobRecord] = {}
        self._active_job_id: UUID | None = None
        self._sync_active = False
        self._sync_cancel_scope: anyio.CancelScope | None = None
        self._sync_done = anyio.Event()
        self._sync_done.set()
        self._accepting = True

    def start(self, job_id: UUID, request: TranscriptionRequest) -> TranscriptionJobStatus:
        """Accept one job or return the retained idempotent result."""
        existing = self._jobs.get(job_id)
        if existing is not None:
            if existing.request != request:
                raise JobConflictError(job_id=job_id)
            return existing.status
        self._reserve(job_id)
        created_at = _now()
        progress = JobProgress(
            phase=TranscriptionPhase.QUEUED,
            completed_chunks=0,
            message="Queued for transcription",
        )
        record = _JobRecord(
            request=request,
            status=QueuedJob(job_id=str(job_id), created_at=created_at, progress=progress),
            done=anyio.Event(),
        )
        self._jobs[job_id] = record
        _ = self._task_group.start_soon(self._run, job_id, record)
        return record.status

    def status(self, job_id: UUID) -> TranscriptionJobStatus:
        """Return the latest retained status for one job."""
        record = self._jobs.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id=job_id)
        return record.status

    def cancel(self, job_id: UUID) -> TranscriptionJobStatus:
        """Request cancellation without waiting for process cleanup."""
        record = self._jobs.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id=job_id)
        match record.status:  # noqa: MATCH_OK - all six status variants are handled
            case QueuedJob(created_at=created_at, progress=progress):
                started_at = None
            case RunningJob(created_at=created_at, started_at=started_at, progress=progress):
                pass
            case CancellingJob() | CompletedJob() | FailedJob() | CancelledJob():
                return record.status
        record.status = CancellingJob(
            job_id=str(job_id),
            created_at=created_at,
            started_at=started_at,
            progress=progress,
        )
        if record.cancel_scope is not None:
            record.cancel_scope.cancel()
        return record.status

    async def wait(self, job_id: UUID) -> TranscriptionJobStatus:
        """Wait until one retained job reaches a terminal state."""
        record = self._jobs.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id=job_id)
        await record.done.wait()
        return record.status

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Run the compatibility synchronous API through shared admission."""
        if not self._accepting or self._sync_active or self._active_job_id is not None:
            raise TranscriptionBusyError
        self._sync_active = True
        self._sync_done = anyio.Event()
        scope = anyio.CancelScope()
        self._sync_cancel_scope = scope
        try:
            with scope:
                return await self._service.transcribe(request)
            raise anyio.get_cancelled_exc_class()
        finally:
            self._sync_cancel_scope = None
            self._sync_active = False
            self._sync_done.set()

    async def aclose(self) -> None:
        """Stop admission and await active job cleanup."""
        self._accepting = False
        if self._sync_active:
            if self._sync_cancel_scope is not None:
                self._sync_cancel_scope.cancel()
            await self._sync_done.wait()
        active_job_id = self._active_job_id
        if active_job_id is None:
            return
        _ = self.cancel(active_job_id)
        _ = await self.wait(active_job_id)

    def _reserve(self, job_id: UUID) -> None:
        if not self._accepting or self._sync_active or self._active_job_id is not None:
            raise TranscriptionBusyError
        self._active_job_id = job_id

    async def _run(self, job_id: UUID, record: _JobRecord) -> None:
        started_at = _now()
        if isinstance(record.status, CancellingJob):
            self._finish_cancelled(job_id, record, started_at=None)
            return
        scope = anyio.CancelScope()
        record.cancel_scope = scope
        record.status = RunningJob(
            job_id=str(job_id),
            created_at=record.status.created_at,
            started_at=started_at,
            progress=JobProgress(
                phase=TranscriptionPhase.STARTING_ENGINE,
                completed_chunks=0,
                message="Starting speech engine",
            ),
        )
        with scope:
            try:
                result = await self._service.transcribe(
                    record.request,
                    self._reporter(record),
                )
            except anyio.get_cancelled_exc_class():
                self._finish_cancelled(job_id, record, started_at=started_at)
                raise
            except Exception as error:  # noqa: BROAD_EXCEPT_OK
                progress = record.status.progress
                record.status = FailedJob(
                    job_id=str(job_id),
                    created_at=record.status.created_at,
                    started_at=started_at,
                    finished_at=_now(),
                    progress=progress,
                    error=JobError(phase=progress.phase, message=str(error)),
                )
                LOGGER.exception("transcription job failed", extra={"job_id": str(job_id)})
            else:
                progress = record.status.progress
                record.status = CompletedJob(
                    job_id=str(job_id),
                    created_at=record.status.created_at,
                    started_at=started_at,
                    finished_at=_now(),
                    progress=progress,
                    result=result,
                )
        if scope.cancel_called:
            if not isinstance(record.status, CancelledJob):
                self._finish_cancelled(job_id, record, started_at=started_at)
            return
        self._release(job_id, record)

    def _reporter(
        self,
        record: _JobRecord,
    ) -> ProgressReporter:
        async def report(
            phase: str,
            completed_chunks: int,
            total_chunks: int | None,
            message: str,
            /,
        ) -> None:
            current = record.status
            if isinstance(current, RunningJob):
                record.status = current.model_copy(
                    update={
                        "progress": JobProgress(
                            phase=TranscriptionPhase(phase),
                            completed_chunks=completed_chunks,
                            total_chunks=total_chunks,
                            message=message,
                        )
                    }
                )

        return report

    def _finish_cancelled(
        self,
        job_id: UUID,
        record: _JobRecord,
        *,
        started_at: datetime | None,
    ) -> None:
        current = record.status
        record.status = CancelledJob(
            job_id=str(job_id),
            created_at=current.created_at,
            started_at=started_at,
            finished_at=_now(),
            progress=current.progress,
        )
        self._release(job_id, record)

    def _release(self, job_id: UUID, record: _JobRecord) -> None:
        if self._active_job_id == job_id:
            self._active_job_id = None
        record.done.set()


def _now() -> datetime:
    return datetime.now(tz=UTC)
