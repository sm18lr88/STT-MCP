from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final
from uuid import UUID, uuid4

import anyio
import pytest

from stt_mcp.backend import Backend, TranscriptionBusyError
from stt_mcp.contracts import ArtifactFormat, TimingQuality, TranscriptDocument
from stt_mcp.job_contracts import CancelledJob, CancellingJob, FailedJob, RunningJob
from stt_mcp.job_support import JobConflictError
from stt_mcp.jobs import TranscriptionJobManager
from stt_mcp.service import (
    ProgressReporter,
    TranscriptionRequest,
    TranscriptionResult,
)

if TYPE_CHECKING:
    from pathlib import Path


@final
class BlockingService:
    """Controllable transcriber for job lifecycle tests."""

    def __init__(self) -> None:
        self.entered = anyio.Event()
        self.release = anyio.Event()
        self.cleanup_entered = anyio.Event()
        self.cleanup_release = anyio.Event()

    async def transcribe(
        self,
        request: TranscriptionRequest,
        progress: ProgressReporter | None = None,
    ) -> TranscriptionResult:
        if progress is not None:
            await progress("normalizing", 0, None, "Normalizing source media")
        self.entered.set()
        try:
            await self.release.wait()
        except anyio.get_cancelled_exc_class():
            self.cleanup_entered.set()
            with anyio.CancelScope(shield=True):
                await self.cleanup_release.wait()
            raise
        return TranscriptionResult(
            document=TranscriptDocument(
                source_path=request.source_path,
                backend=Backend.PARAKEET,
                duration_seconds=1.0,
                timing_quality=TimingQuality.COARSE_SOURCE_WINDOW,
                segments=(),
            ),
            artifacts=(),
        )


@dataclass(frozen=True, slots=True)
class FailingService:
    """Transcriber that exposes one terminal infrastructure failure."""

    async def transcribe(
        self,
        request: TranscriptionRequest,
        progress: ProgressReporter | None = None,
    ) -> TranscriptionResult:
        _ = request
        if progress is not None:
            await progress("starting_engine", 0, None, "Starting speech engine")
        msg = "model initialization failed"
        raise RuntimeError(msg)


def _request(tmp_path: Path, *, source_name: str = "speech.wav") -> TranscriptionRequest:
    return TranscriptionRequest(
        source_path=tmp_path / source_name,
        output_directory=tmp_path,
        formats=(ArtifactFormat.TEXT,),
    )


@pytest.mark.anyio
async def test_start_returns_before_blocked_transcription_finishes(tmp_path: Path) -> None:
    # Given
    service = BlockingService()
    job_id = uuid4()

    async with anyio.create_task_group() as tasks:
        manager = TranscriptionJobManager(service, tasks)

        # When
        accepted = manager.start(job_id, _request(tmp_path))
        await service.entered.wait()
        current = manager.status(job_id)

        # Then
        assert accepted.status == "queued"
        assert isinstance(current, RunningJob)
        assert current.progress.phase == "normalizing"

        service.release.set()
        await manager.aclose()


@pytest.mark.anyio
async def test_start_is_idempotent_and_rejects_conflicting_inputs(tmp_path: Path) -> None:
    # Given
    service = BlockingService()
    job_id = uuid4()
    request = _request(tmp_path)

    async with anyio.create_task_group() as tasks:
        manager = TranscriptionJobManager(service, tasks)
        original = manager.start(job_id, request)

        # When / Then
        assert manager.start(job_id, request) == original
        with pytest.raises(JobConflictError):
            _ = manager.start(job_id, _request(tmp_path, source_name="other.wav"))
        with pytest.raises(TranscriptionBusyError):
            _ = manager.start(uuid4(), request)

        service.release.set()
        await manager.aclose()


@pytest.mark.anyio
async def test_failure_is_retained_as_terminal_status(tmp_path: Path) -> None:
    # Given
    job_id = uuid4()

    async with anyio.create_task_group() as tasks:
        manager = TranscriptionJobManager(FailingService(), tasks)

        # When
        _ = manager.start(job_id, _request(tmp_path))
        status = await manager.wait(job_id)

        # Then
        assert isinstance(status, FailedJob)
        assert status.error.phase == "starting_engine"
        assert status.error.message == "model initialization failed"


@pytest.mark.anyio
async def test_terminal_job_identity_is_retained_for_process_lifetime(tmp_path: Path) -> None:
    # Given
    first_job_id = uuid4()
    first_request = _request(tmp_path)

    async with anyio.create_task_group() as tasks:
        manager = TranscriptionJobManager(FailingService(), tasks)

        # When
        _ = manager.start(first_job_id, first_request)
        first_status = await manager.wait(first_job_id)
        for index in range(33):
            job_id = uuid4()
            _ = manager.start(job_id, _request(tmp_path, source_name=f"speech-{index}.wav"))
            _ = await manager.wait(job_id)

        # Then
        assert manager.status(first_job_id) == first_status
        assert manager.start(first_job_id, first_request) == first_status
        with pytest.raises(JobConflictError):
            _ = manager.start(first_job_id, _request(tmp_path, source_name="other.wav"))


@pytest.mark.anyio
async def test_cancellation_keeps_admission_until_cleanup_finishes(tmp_path: Path) -> None:
    # Given
    service = BlockingService()
    job_id = UUID("27ac63c1-e0d7-4aa8-bca9-3c4910ea4e31")

    async with anyio.create_task_group() as tasks:
        manager = TranscriptionJobManager(service, tasks)
        _ = manager.start(job_id, _request(tmp_path))
        await service.entered.wait()

        # When
        cancelling = manager.cancel(job_id)
        await service.cleanup_entered.wait()

        # Then
        assert isinstance(cancelling, CancellingJob)
        with pytest.raises(TranscriptionBusyError):
            _ = manager.start(uuid4(), _request(tmp_path))

        service.cleanup_release.set()
        cancelled = await manager.wait(job_id)
        assert isinstance(cancelled, CancelledJob)

        next_job = uuid4()
        accepted = manager.start(next_job, _request(tmp_path))
        assert accepted.status == "queued"
        service.release.set()
        await manager.aclose()


@pytest.mark.anyio
async def test_external_task_group_cancellation_reaches_job_task(tmp_path: Path) -> None:
    # Given
    service = BlockingService()
    job_id = uuid4()
    managers: list[TranscriptionJobManager] = []

    # When
    async with anyio.create_task_group() as tasks:
        manager = TranscriptionJobManager(service, tasks)
        managers.append(manager)
        _ = manager.start(job_id, _request(tmp_path))
        await service.entered.wait()
        service.cleanup_release.set()
        tasks.cancel_scope.cancel()

    # Then
    assert len(managers) == 1
    assert isinstance(managers[0].status(job_id), CancelledJob)


@pytest.mark.anyio
async def test_combined_job_and_parent_cancellation_finishes_terminal_state(tmp_path: Path) -> None:
    # Given
    service = BlockingService()
    job_id = uuid4()
    managers: list[TranscriptionJobManager] = []

    # When
    async with anyio.create_task_group() as tasks:
        manager = TranscriptionJobManager(service, tasks)
        managers.append(manager)
        _ = manager.start(job_id, _request(tmp_path))
        await service.entered.wait()
        _ = manager.cancel(job_id)
        await service.cleanup_entered.wait()
        service.cleanup_release.set()
        tasks.cancel_scope.cancel()

    # Then
    assert len(managers) == 1
    assert isinstance(managers[0].status(job_id), CancelledJob)
    with anyio.fail_after(1):
        assert isinstance(await managers[0].wait(job_id), CancelledJob)


@pytest.mark.anyio
async def test_close_cancels_and_drains_synchronous_transcription(tmp_path: Path) -> None:
    # Given
    service = BlockingService()

    async with anyio.create_task_group() as tasks:
        manager = TranscriptionJobManager(service, tasks)
        _ = tasks.start_soon(manager.transcribe, _request(tmp_path))
        await service.entered.wait()

        # When
        _ = tasks.start_soon(manager.aclose)
        await service.cleanup_entered.wait()
        with pytest.raises(TranscriptionBusyError):
            _ = manager.start(uuid4(), _request(tmp_path))
        service.cleanup_release.set()
