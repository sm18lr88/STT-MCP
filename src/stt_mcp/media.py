"""Cancellable FFmpeg media probing and audio normalization."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, override

import anyio
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from stt_mcp.transcript import plan_source_windows

CREATE_NO_WINDOW: Final = 0x08000000


class _ProbeStream(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)


class _ProbeFormat(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    duration: float = Field(gt=0.0)


class _ProbePayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    streams: tuple[_ProbeStream, ...]
    format: _ProbeFormat


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Validated properties of the first audio stream."""

    duration_seconds: float
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class NormalizedChunk:
    """One normalized WAV chunk and its coarse source timing."""

    index: int
    path: Path
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class MediaToolError(Exception):
    """Raised when FFmpeg or FFprobe cannot process source media."""

    executable: str
    source_path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.executable} could not process {self.source_path}: {self.detail}"


async def probe_media(source_path: Path) -> MediaInfo:
    """Probe the first audio stream and parse its media properties."""
    if not await anyio.Path(source_path).is_file():
        raise MediaToolError(
            executable="ffprobe",
            source_path=source_path,
            detail="source file does not exist",
        )

    completed = await anyio.run_process(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration:stream=sample_rate,channels",
            "-of",
            "json",
            str(source_path),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        creationflags=_creation_flags(),
    )
    if completed.returncode != 0:
        raise MediaToolError(
            executable="ffprobe",
            source_path=source_path,
            detail=completed.stderr.decode("utf-8", errors="replace").strip(),
        )

    try:
        payload = _ProbePayload.model_validate_json(completed.stdout)
    except ValidationError as error:
        raise MediaToolError(
            executable="ffprobe",
            source_path=source_path,
            detail="media has no readable audio stream or duration",
        ) from error
    if not payload.streams:
        raise MediaToolError(
            executable="ffprobe",
            source_path=source_path,
            detail="media has no readable audio stream",
        )
    stream = payload.streams[0]
    return MediaInfo(
        duration_seconds=payload.format.duration,
        sample_rate=stream.sample_rate,
        channels=stream.channels,
    )


async def normalize_media(
    *, source_path: Path, output_directory: Path
) -> tuple[NormalizedChunk, ...]:
    """Decode source media into 30-second 16 kHz mono PCM WAV chunks."""
    info = await probe_media(source_path)
    async_output_directory = anyio.Path(output_directory)
    await async_output_directory.mkdir(parents=True, exist_ok=True)
    output_pattern = output_directory / "chunk-%06d.wav"
    completed = await anyio.run_process(
        (
            "ffmpeg",
            "-xerror",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "segment",
            "-segment_time",
            "30",
            "-reset_timestamps",
            "1",
            str(output_pattern),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        creationflags=_creation_flags(),
    )
    if completed.returncode != 0:
        raise MediaToolError(
            executable="ffmpeg",
            source_path=source_path,
            detail=completed.stderr.decode("utf-8", errors="replace").strip(),
        )

    chunk_paths = tuple(
        sorted([Path(path) async for path in async_output_directory.glob("chunk-*.wav")])
    )
    windows = plan_source_windows(duration_seconds=info.duration_seconds)
    if len(chunk_paths) != len(windows):
        raise MediaToolError(
            executable="ffmpeg",
            source_path=source_path,
            detail=f"expected {len(windows)} chunks but decoded {len(chunk_paths)}",
        )
    return tuple(
        NormalizedChunk(
            index=window.index,
            path=chunk_path,
            start_seconds=window.start_seconds,
            end_seconds=window.end_seconds,
        )
        for chunk_path, window in zip(chunk_paths, windows, strict=True)
    )


def _creation_flags() -> int:
    return CREATE_NO_WINDOW if sys.platform == "win32" else 0
