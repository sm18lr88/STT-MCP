from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from stt_mcp.backend import Backend
from stt_mcp.server import mcp
from stt_mcp.service import TranscriptionResult

REFERENCE_MEDIA_PATH = Path(__file__).parent.parent / "assets" / "test-audio.mp3"
REFERENCE_MEDIA_SHA256 = "dfb6ef4cc9ad03ba54e24026ab734a56bf7e3251751e634e88f6f384402bff45"


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
