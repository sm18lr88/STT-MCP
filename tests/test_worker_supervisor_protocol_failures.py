from __future__ import annotations

from typing import TYPE_CHECKING, override

import anyio
import pytest

from stt_mcp.media import NormalizedChunk
from stt_mcp.runtime_policy import select_runtime
from stt_mcp.supervisor import WorkerFailureError, WorkerSupervisor
from stt_mcp.worker_protocol import WorkerCommand, WorkerProtocolError

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


class BlockingDiagnosticProcess(FakeProcess):
    wait_started: anyio.Event
    wait_count: int

    def __init__(self, wait_started: anyio.Event) -> None:
        super().__init__()
        self.wait_started = wait_started

    @override
    async def wait(self) -> int:
        self.wait_count += 1
        if self.wait_count == 1:
            self.wait_started.set()
            await anyio.sleep_forever()
        return self.returncode or 0


class OversizedReader:
    async def receive_until(self, delimiter: bytes, max_bytes: int) -> bytes:
        assert delimiter == b"\n"
        raise anyio.DelimiterNotFound(max_bytes)


class IncompleteReader:
    async def receive_until(self, delimiter: bytes, max_bytes: int) -> bytes:
        assert delimiter == b"\n"
        assert max_bytes > 0
        raise anyio.IncompleteRead


def normalized_chunk(tmp_path: Path) -> NormalizedChunk:
    return NormalizedChunk(
        index=2,
        path=tmp_path / "chunk.wav",
        start_seconds=30.0,
        end_seconds=60.0,
    )


@pytest.mark.parametrize(
    "communication_error",
    [anyio.BrokenResourceError, WorkerProtocolError(detail="malformed ready event")],
)
@pytest.mark.anyio
async def test_start_normalizes_communication_failure(
    monkeypatch: pytest.MonkeyPatch,
    communication_error: type[anyio.BrokenResourceError] | WorkerProtocolError,
) -> None:
    supervisor = WorkerSupervisor(select_runtime(platform="win32", machine="AMD64"))
    process = FakeProcess()

    async def fail_start(_supervisor: WorkerSupervisor) -> None:
        if isinstance(communication_error, type):
            raise communication_error
        raise communication_error

    monkeypatch.setattr(supervisor, "_process", process)
    monkeypatch.setattr(WorkerSupervisor, "_ensure_started", fail_start)

    with pytest.raises(WorkerFailureError, match="worker communication failed"):
        await supervisor.start()

    assert process.kill_count == 1
    assert process.wait_count == 1
    assert process.close_count == 1


@pytest.mark.anyio
async def test_transcribe_invalidates_worker_after_oversized_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = WorkerSupervisor(select_runtime(platform="win32", machine="AMD64"))
    process = FakeProcess()

    async def fake_ensure_started(_supervisor: WorkerSupervisor) -> None:
        return

    async def fake_send(_supervisor: WorkerSupervisor, _command: WorkerCommand) -> None:
        return

    monkeypatch.setattr(supervisor, "_process", process)
    monkeypatch.setattr(supervisor, "_reader", OversizedReader())
    monkeypatch.setattr(WorkerSupervisor, "_ensure_started", fake_ensure_started)
    monkeypatch.setattr(WorkerSupervisor, "_send", fake_send)

    with pytest.raises(WorkerFailureError, match="worker protocol line exceeded"):
        _ = await supervisor.transcribe(normalized_chunk(tmp_path))

    assert process.kill_count == 1
    assert process.wait_count == 1
    assert process.close_count == 1


@pytest.mark.anyio
async def test_closed_worker_diagnostic_cancellation_finishes_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = WorkerSupervisor(select_runtime(platform="win32", machine="AMD64"))
    wait_started = anyio.Event()
    process = BlockingDiagnosticProcess(wait_started)

    async def fake_ensure_started(_supervisor: WorkerSupervisor) -> None:
        return

    async def fake_send(_supervisor: WorkerSupervisor, _command: WorkerCommand) -> None:
        return

    monkeypatch.setattr(supervisor, "_process", process)
    monkeypatch.setattr(supervisor, "_reader", IncompleteReader())
    monkeypatch.setattr(WorkerSupervisor, "_ensure_started", fake_ensure_started)
    monkeypatch.setattr(WorkerSupervisor, "_send", fake_send)

    async with anyio.create_task_group() as tasks:
        _ = tasks.start_soon(supervisor.transcribe, normalized_chunk(tmp_path))
        await wait_started.wait()
        tasks.cancel_scope.cancel()

    assert process.kill_count == 1
    assert process.wait_count == 2
    assert process.close_count == 1
