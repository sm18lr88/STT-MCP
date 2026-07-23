from __future__ import annotations

import os
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from stt_mcp.backend import Backend
from stt_mcp.server import mcp
from stt_mcp.service import TranscriptionResult


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
