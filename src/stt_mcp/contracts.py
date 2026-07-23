"""Validated transcript and artifact contracts."""

from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from stt_mcp.backend import Backend


class ArtifactFormat(StrEnum):
    """Supported transcript artifact formats."""

    TEXT = "txt"
    MARKDOWN = "md"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"


class TimingQuality(StrEnum):
    """Provenance quality of transcript timing information."""

    COARSE_SOURCE_WINDOW = "coarse_source_window"


class TranscriptSegment(BaseModel):
    """One source-window transcript segment."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    text: str


class TranscriptDocument(BaseModel):
    """Complete transcript and its timing provenance."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source_path: Path
    backend: Backend
    duration_seconds: float = Field(gt=0.0)
    timing_quality: TimingQuality
    segments: tuple[TranscriptSegment, ...]

    @property
    def text(self) -> str:
        """Return normalized full transcript text."""
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())


class ArtifactRecord(BaseModel):
    """One published transcript artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    format: ArtifactFormat
    path: Path


class TranscriptJsonPayload(BaseModel):
    """Machine-readable JSON transcript representation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source_path: Path
    backend: Backend
    duration_seconds: float
    timing_quality: TimingQuality
    segments: tuple[TranscriptSegment, ...]
    text: str
