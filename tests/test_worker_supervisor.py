from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from stt_mcp.media import NormalizedChunk
from stt_mcp.runtime_policy import select_runtime
from stt_mcp.supervisor import WorkerFailureError, WorkerSupervisor
from stt_mcp.worker_protocol import (
    ErrorEvent,
    ResultEvent,
    ShutdownCompleteEvent,
    TranscribeCommand,
    WorkerErrorCode,
)

if TYPE_CHECKING:
    from pathlib import Path

    from stt_mcp.worker_protocol import WorkerCommand


@pytest.mark.anyio
async def test_transcribe_invalidates_worker_after_unexpected_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    supervisor = WorkerSupervisor(select_runtime(platform="win32", machine="AMD64"))
    invalidations: list[bool] = []

    async def fake_ensure_started(_supervisor: WorkerSupervisor) -> None:
        return

    async def fake_send(_supervisor: WorkerSupervisor, _command: WorkerCommand) -> None:
        return

    async def fake_receive(_supervisor: WorkerSupervisor) -> ShutdownCompleteEvent:
        return ShutdownCompleteEvent()

    async def fake_invalidate(_supervisor: WorkerSupervisor) -> None:
        invalidations.append(True)

    monkeypatch.setattr(WorkerSupervisor, "_ensure_started", fake_ensure_started)
    monkeypatch.setattr(WorkerSupervisor, "_send", fake_send)
    monkeypatch.setattr(WorkerSupervisor, "_receive", fake_receive)
    monkeypatch.setattr(WorkerSupervisor, "_invalidate", fake_invalidate)
    chunk = NormalizedChunk(
        index=4,
        path=tmp_path / "audio" / "chunk.wav",
        start_seconds=120.0,
        end_seconds=150.0,
    )

    # When / Then
    with pytest.raises(WorkerFailureError, match="unexpected event"):
        _ = await supervisor.transcribe(chunk)
    assert invalidations == [True]


@pytest.mark.anyio
async def test_transcribe_returns_matching_worker_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = WorkerSupervisor(select_runtime(platform="win32", machine="AMD64"))
    commands: list[WorkerCommand] = []

    async def fake_ensure_started(_supervisor: WorkerSupervisor) -> None:
        return

    async def fake_send(_supervisor: WorkerSupervisor, command: WorkerCommand) -> None:
        commands.append(command)

    async def fake_receive(_supervisor: WorkerSupervisor) -> ResultEvent:
        command = commands[-1]
        assert isinstance(command, TranscribeCommand)
        return ResultEvent(
            request_id=command.request_id,
            chunk_index=command.chunk_index,
            text="matched transcription",
        )

    monkeypatch.setattr(WorkerSupervisor, "_ensure_started", fake_ensure_started)
    monkeypatch.setattr(WorkerSupervisor, "_send", fake_send)
    monkeypatch.setattr(WorkerSupervisor, "_receive", fake_receive)
    chunk = NormalizedChunk(
        index=6,
        path=tmp_path / "audio" / "chunk.wav",
        start_seconds=180.0,
        end_seconds=210.0,
    )

    text = await supervisor.transcribe(chunk)

    assert text == "matched transcription"


@pytest.mark.parametrize(
    ("reusability", "expected_invalidations"),
    [("reusable", 0), ("terminal", 1)],
)
@pytest.mark.anyio
async def test_transcribe_propagates_worker_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reusability: str,
    expected_invalidations: int,
) -> None:
    supervisor = WorkerSupervisor(select_runtime(platform="win32", machine="AMD64"))
    commands: list[WorkerCommand] = []
    invalidations: list[bool] = []

    async def fake_ensure_started(_supervisor: WorkerSupervisor) -> None:
        return

    async def fake_send(_supervisor: WorkerSupervisor, command: WorkerCommand) -> None:
        commands.append(command)

    async def fake_receive(_supervisor: WorkerSupervisor) -> ErrorEvent:
        command = commands[-1]
        assert isinstance(command, TranscribeCommand)
        return ErrorEvent(
            request_id=command.request_id,
            chunk_index=command.chunk_index,
            code=WorkerErrorCode.INFERENCE_FAILED,
            message="worker inference failed",
            worker_reusable=reusability == "reusable",
        )

    async def fake_invalidate(_supervisor: WorkerSupervisor) -> None:
        invalidations.append(True)

    monkeypatch.setattr(WorkerSupervisor, "_ensure_started", fake_ensure_started)
    monkeypatch.setattr(WorkerSupervisor, "_send", fake_send)
    monkeypatch.setattr(WorkerSupervisor, "_receive", fake_receive)
    monkeypatch.setattr(WorkerSupervisor, "_invalidate", fake_invalidate)
    chunk = NormalizedChunk(
        index=7,
        path=tmp_path / "audio" / "chunk.wav",
        start_seconds=210.0,
        end_seconds=240.0,
    )

    with pytest.raises(WorkerFailureError, match="worker inference failed"):
        _ = await supervisor.transcribe(chunk)

    assert len(invalidations) == expected_invalidations
