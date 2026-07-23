from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import anyio
import pytest

from stt_mcp.media import MediaInfo, normalize_media, probe_media

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


@pytest.mark.anyio
async def test_normalize_media_exits_on_ffmpeg_decode_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_ffmpeg(
        command: tuple[str, ...],
        *,
        stdout: int,
        stderr: int,
        check: bool,
        creationflags: int,
    ) -> subprocess.CompletedProcess[bytes]:
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE
        assert check is False
        assert creationflags >= 0
        assert "-xerror" in command
        await anyio.Path(command[-1].replace("%06d", "000000")).touch()
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    async def probe_damaged_media(_: Path) -> MediaInfo:
        return MediaInfo(duration_seconds=1.0, sample_rate=16_000, channels=1)

    # Given
    monkeypatch.setattr("stt_mcp.media.anyio.run_process", run_ffmpeg)
    monkeypatch.setattr(
        "stt_mcp.media.probe_media",
        probe_damaged_media,
    )

    # When
    chunks = await normalize_media(
        source_path=tmp_path / "damaged.mp4",
        output_directory=tmp_path / "chunks",
    )

    # Then
    assert len(chunks) == 1
