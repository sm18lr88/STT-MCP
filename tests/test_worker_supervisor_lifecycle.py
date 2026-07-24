from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio
import pytest

from stt_mcp.media import NormalizedChunk
from stt_mcp.runtime_policy import select_runtime
from stt_mcp.supervisor import WorkerBusyError, WorkerFailureError, WorkerSupervisor
from stt_mcp.worker_protocol import ReadyEvent, ResultEvent, WorkerCommand, WorkerProtocolError

if TYPE_CHECKING:
    from pathlib import Path


class FakeProcess:
    returncode: int | None
    kill_count: int
    wait_count: int
    close_count: int

    def __init__(self) -> None:
        self.returncode = None
        self.kill_count = 0
        self.wait_count = 0
        self.close_count = 0

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_count += 1
        return self.returncode or 0

    async def aclose(self) -> None:
        self.close_count += 1


class FakeByteReceiveStream:
    pass


@dataclass(frozen=True, slots=True)
class FakeOpenWorker:
    process: FakeProcess
    stdout: FakeByteReceiveStream = field(default_factory=FakeByteReceiveStream)
    stderr: FakeByteReceiveStream = field(default_factory=FakeByteReceiveStream)


def normalized_chunk(tmp_path: Path) -> NormalizedChunk:
    return NormalizedChunk(
        index=1,
        path=tmp_path / "chunk.wav",
        start_seconds=0.0,
        end_seconds=30.0,
    )


@pytest.mark.anyio
async def test_start_cancellation_closes_partially_initialized_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = WorkerSupervisor(select_runtime(platform="win32", machine="AMD64"))
    process = FakeProcess()
    receiving = anyio.Event()

    async def fake_open_process(_supervisor: WorkerSupervisor) -> FakeOpenWorker:
        return FakeOpenWorker(process=process)

    async def fake_send(_supervisor: WorkerSupervisor, _command: WorkerCommand) -> None:
        return

    async def fake_receive(_supervisor: WorkerSupervisor) -> ReadyEvent:
        receiving.set()
        await anyio.sleep_forever()
        raise AssertionError

    monkeypatch.setattr(WorkerSupervisor, "_open_process", fake_open_process)
    monkeypatch.setattr(WorkerSupervisor, "_send", fake_send)
    monkeypatch.setattr(WorkerSupervisor, "_receive", fake_receive)

    async with anyio.create_task_group() as tasks:
        _ = tasks.start_soon(supervisor.start)
        await receiving.wait()
        tasks.cancel_scope.cancel()

    assert process.kill_count == 1
    assert process.wait_count == 1
    assert process.close_count == 1


@pytest.mark.anyio
async def test_start_rejects_concurrent_owner_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = WorkerSupervisor(select_runtime(platform="win32", machine="AMD64"))
    started = anyio.Event()
    release = anyio.Event()

    async def fake_ensure_started(_supervisor: WorkerSupervisor) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(WorkerSupervisor, "_ensure_started", fake_ensure_started)

    async with anyio.create_task_group() as tasks:
        _ = tasks.start_soon(supervisor.start)
        await started.wait()
        with pytest.raises(WorkerBusyError):
            await supervisor.start()
        release.set()


@pytest.mark.parametrize(
    "communication_error",
    [
        anyio.BrokenResourceError,
        WorkerProtocolError(detail="malformed worker event"),
    ],
)
@pytest.mark.anyio
async def test_transcribe_invalidates_worker_after_communication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    communication_error: type[anyio.BrokenResourceError] | WorkerProtocolError,
) -> None:
    supervisor = WorkerSupervisor(select_runtime(platform="win32", machine="AMD64"))
    process = FakeProcess()

    async def fake_ensure_started(_supervisor: WorkerSupervisor) -> None:
        return

    async def fake_send(_supervisor: WorkerSupervisor, _command: WorkerCommand) -> None:
        if isinstance(communication_error, type):
            raise communication_error

    async def fake_receive(_supervisor: WorkerSupervisor) -> ResultEvent:
        if isinstance(communication_error, WorkerProtocolError):
            raise communication_error
        raise AssertionError

    monkeypatch.setattr(supervisor, "_process", process)
    monkeypatch.setattr(WorkerSupervisor, "_ensure_started", fake_ensure_started)
    monkeypatch.setattr(WorkerSupervisor, "_send", fake_send)
    monkeypatch.setattr(WorkerSupervisor, "_receive", fake_receive)

    with pytest.raises(WorkerFailureError, match="worker communication failed"):
        _ = await supervisor.transcribe(normalized_chunk(tmp_path))

    assert process.kill_count == 1
    assert process.wait_count == 1
    assert process.close_count == 1


@pytest.mark.anyio
async def test_close_invalidates_worker_after_malformed_shutdown_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = WorkerSupervisor(select_runtime(platform="win32", machine="AMD64"))
    process = FakeProcess()

    async def fake_send(_supervisor: WorkerSupervisor, _command: WorkerCommand) -> None:
        return

    async def fake_receive(_supervisor: WorkerSupervisor) -> ReadyEvent:
        raise WorkerProtocolError(detail="malformed shutdown event")

    monkeypatch.setattr(supervisor, "_process", process)
    monkeypatch.setattr(WorkerSupervisor, "_send", fake_send)
    monkeypatch.setattr(WorkerSupervisor, "_receive", fake_receive)

    await supervisor.aclose()
    await supervisor.aclose()

    assert process.kill_count == 1
    assert process.wait_count == 1
    assert process.close_count == 1
