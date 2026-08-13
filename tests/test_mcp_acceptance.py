from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, final, override
from uuid import uuid4

import anyio
import pytest
from anyio.lowlevel import checkpoint
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent
from pydantic import Field, TypeAdapter

import stt_mcp.server as server_module
from stt_mcp.backend import AudioChunk, Backend
from stt_mcp.job_contracts import (
    CancelledJob,
    CancellingJob,
    CompletedJob,
    FailedJob,
    QueuedJob,
    RunningJob,
)
from stt_mcp.server import mcp
from stt_mcp.service import TranscriptionResult

if TYPE_CHECKING:
    from mcp.client.session import ClientSession

REFERENCE_MEDIA_PATH = Path(__file__).parent.parent / "assets" / "test-audio.mp3"
REFERENCE_MEDIA_SHA256 = "dfb6ef4cc9ad03ba54e24026ab734a56bf7e3251751e634e88f6f384402bff45"
type ConcreteJobStatus = (
    QueuedJob | RunningJob | CancellingJob | CompletedJob | FailedJob | CancelledJob
)
JOB_STATUS_ADAPTER: TypeAdapter[ConcreteJobStatus] = TypeAdapter(
    Annotated[ConcreteJobStatus, Field(discriminator="status")]
)


class ControlledEngine:
    mode: str
    entered: anyio.Event
    cleanup_entered: anyio.Event
    cleanup_release: anyio.Event

    def __init__(self) -> None:
        self.mode = "complete"
        self.entered = anyio.Event()
        self.cleanup_entered = anyio.Event()
        self.cleanup_release = anyio.Event()

    @property
    def backend(self) -> Backend:
        return Backend.GRANITE

    async def start(self) -> None:
        return

    async def transcribe(self, chunk: AudioChunk) -> str:
        self.entered.set()
        if self.mode == "fail":
            msg = "controlled inference failure"
            raise RuntimeError(msg)
        if self.mode == "block":
            try:
                await anyio.sleep_forever()
            except anyio.get_cancelled_exc_class():
                self.cleanup_entered.set()
                with anyio.CancelScope(shield=True):
                    await self.cleanup_release.wait()
                raise
        return f"controlled chunk {chunk.index + 1}"

    async def aclose(self) -> None:
        return


@final
class ClosingEngine(ControlledEngine):
    def __init__(self) -> None:
        super().__init__()
        self.mode = "block"
        self.closed = anyio.Event()

    @override
    async def aclose(self) -> None:
        self.closed.set()


async def poll_terminal(session: ClientSession, job_id: str) -> ConcreteJobStatus:
    while True:
        response = await session.call_tool("get_transcription_status", {"job_id": job_id})
        assert response.isError is False
        assert response.structuredContent is not None
        status = JOB_STATUS_ADAPTER.validate_json(
            json.dumps(response.structuredContent["result"])
        )
        if status.status in {"completed", "failed", "cancelled"}:
            return status
        await checkpoint()


def test_reference_media_matches_pinned_checksum() -> None:
    """Keep the checked-in acceptance media stable and reproducible."""
    # Given
    assert REFERENCE_MEDIA_PATH.is_file()

    # When
    with REFERENCE_MEDIA_PATH.open("rb") as media:
        digest = hashlib.file_digest(media, "sha256").hexdigest()

    # Then
    assert digest == REFERENCE_MEDIA_SHA256


def configured_media_path() -> Path:
    configured = os.environ.get("STT_MCP_TEST_MEDIA")
    if configured is None:
        pytest.skip("set STT_MCP_TEST_MEDIA to run the configured backend acceptance test")
    assert configured is not None
    source = Path(configured).expanduser().resolve()
    if not source.is_file():
        pytest.fail(f"STT_MCP_TEST_MEDIA is not a readable file: {source}")
    return source


@pytest.mark.anyio
async def test_mcp_schema_uses_supported_json_schema_formats() -> None:
    # Given / When
    async with create_connected_server_and_client_session(
        mcp,
        raise_exceptions=True,
    ) as session:
        result = await session.list_tools()

    tool = next(item for item in result.tools if item.name == "transcribe")
    schema = json.dumps(
        {"input": tool.inputSchema, "output": tool.outputSchema},
        sort_keys=True,
    )

    # Then
    assert '"format": "path"' not in schema


@pytest.mark.anyio
async def test_mcp_lists_durable_transcription_job_tools() -> None:
    # Given / When
    async with create_connected_server_and_client_session(
        mcp,
        raise_exceptions=True,
    ) as session:
        result = await session.list_tools()

    # Then
    names = {tool.name for tool in result.tools}
    assert {
        "start_transcription",
        "get_transcription_status",
        "cancel_transcription",
    } <= names


@pytest.mark.anyio
async def test_mcp_durable_job_round_trip_and_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    engine = ControlledEngine()
    monkeypatch.setattr(server_module, "create_configured_engine", lambda: engine)
    job_id = str(uuid4())
    arguments = {
        "job_id": job_id,
        "source_path": str(REFERENCE_MEDIA_PATH),
        "output_directory": str(tmp_path),
        "formats": ["txt"],
    }

    async with create_connected_server_and_client_session(mcp, raise_exceptions=True) as session:
        # When
        accepted = await session.call_tool("start_transcription", arguments)
        repeated = await session.call_tool("start_transcription", arguments)
        completed = await poll_terminal(session, job_id)
        conflict = await session.call_tool(
            "start_transcription",
            arguments | {"source_path": str(tmp_path / "other.mp3")},
        )

    # Then
    assert accepted.isError is False
    assert repeated.isError is False
    assert accepted.structuredContent is not None
    assert "result" in accepted.structuredContent
    assert completed.status == "completed"
    assert (tmp_path / "test-audio.txt").is_file()
    assert conflict.isError is True
    conflict_content = conflict.content[0]
    assert isinstance(conflict_content, TextContent)
    assert "different inputs" in conflict_content.text


@pytest.mark.anyio
async def test_mcp_retains_failure_and_finishes_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    engine = ControlledEngine()
    monkeypatch.setattr(server_module, "create_configured_engine", lambda: engine)

    async with create_connected_server_and_client_session(mcp, raise_exceptions=True) as session:
        engine.mode = "fail"
        failed_id = str(uuid4())
        failed_arguments = {
            "job_id": failed_id,
            "source_path": str(REFERENCE_MEDIA_PATH),
            "output_directory": str(tmp_path),
            "formats": ["txt"],
        }
        _ = await session.call_tool("start_transcription", failed_arguments)
        failed = await poll_terminal(session, failed_id)

        engine.mode = "block"
        engine.entered = anyio.Event()
        cancelled_id = str(uuid4())
        _ = await session.call_tool(
            "start_transcription",
            failed_arguments | {"job_id": cancelled_id},
        )
        await engine.entered.wait()
        cancelling = await session.call_tool(
            "cancel_transcription",
            {"job_id": cancelled_id},
        )
        await engine.cleanup_entered.wait()
        engine.cleanup_release.set()
        cancelled = await poll_terminal(session, cancelled_id)

    # Then
    assert failed.status == "failed"
    assert failed.error.message == "controlled inference failure"
    assert cancelling.structuredContent is not None
    cancelling_status = JOB_STATUS_ADAPTER.validate_json(
        json.dumps(cancelling.structuredContent["result"])
    )
    assert cancelling_status.status == "cancelling"
    assert cancelled.status == "cancelled"


@pytest.mark.anyio
async def test_mcp_lifespan_parent_cancellation_closes_active_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    engine = ClosingEngine()
    monkeypatch.setattr(server_module, "create_configured_engine", lambda: engine)
    arguments = {
        "job_id": str(uuid4()),
        "source_path": str(REFERENCE_MEDIA_PATH),
        "output_directory": str(tmp_path),
        "formats": ["txt"],
    }

    # When
    async with anyio.create_task_group() as tasks:
        async def use_server() -> None:
            async with create_connected_server_and_client_session(
                mcp,
                raise_exceptions=True,
                ) as session:
                _ = await session.call_tool("start_transcription", arguments)
                await engine.entered.wait()
                engine.cleanup_release.set()
                tasks.cancel_scope.cancel()

        _ = tasks.start_soon(use_server)

    # Then
    assert engine.closed.is_set()


@pytest.mark.anyio
async def test_mcp_transcribes_configured_media(tmp_path: Path) -> None:
    # Given
    source = configured_media_path()
    arguments: dict[str, str | list[str]] = {
        "source_path": str(source),
        "output_directory": str(tmp_path),
        "formats": ["srt", "vtt"],
    }

    # When
    async with create_connected_server_and_client_session(
        mcp,
        raise_exceptions=True,
    ) as session:
        result = await session.call_tool("transcribe", arguments)

    # Then
    assert result.isError is False
    assert result.structuredContent is not None
    transcription = TranscriptionResult.model_validate(result.structuredContent)
    assert transcription.document.backend in {Backend.GRANITE, Backend.PARAKEET}
    assert (tmp_path / f"{source.stem}.srt").is_file()
    assert (tmp_path / f"{source.stem}.vtt").is_file()
