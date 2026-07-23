from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import anyio
import pytest

from stt_mcp.media import normalize_media, probe_media

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.anyio
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
async def test_normalize_media_produces_16khz_mono_wav(tmp_path: Path) -> None:
    # Given
    source = tmp_path / "source tone.mp4"
    generated = await anyio.run_process(
        (
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:a",
            "aac",
            str(source),
        ),
        check=False,
    )
    assert generated.returncode == 0

    # When
    info = await probe_media(source)
    chunks = await normalize_media(source_path=source, output_directory=tmp_path / "chunks")
    chunk_info = await probe_media(chunks[0].path)

    # Then
    assert 1.9 < info.duration_seconds < 2.1
    assert len(chunks) == 1
    assert chunk_info.sample_rate == 16_000
    assert chunk_info.channels == 1
    assert chunks[0].start_seconds == 0.0
    assert 1.9 < chunks[0].end_seconds < 2.1
