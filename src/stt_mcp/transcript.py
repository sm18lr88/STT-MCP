"""Source-window planning and transcript rendering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Final

from stt_mcp.contracts import ArtifactFormat, TranscriptDocument, TranscriptJsonPayload

WINDOW_SECONDS: Final = 30.0


@dataclass(frozen=True, slots=True)
class SourceWindow:
    """A coarse source-time interval sent to the speech model."""

    index: int
    start_seconds: float
    end_seconds: float


def plan_source_windows(*, duration_seconds: float) -> tuple[SourceWindow, ...]:
    """Split media duration into fixed 30-second source windows."""
    count = math.ceil(duration_seconds / WINDOW_SECONDS)
    return tuple(
        SourceWindow(
            index=index,
            start_seconds=index * WINDOW_SECONDS,
            end_seconds=min((index + 1) * WINDOW_SECONDS, duration_seconds),
        )
        for index in range(count)
    )


def render_transcript(document: TranscriptDocument, artifact_format: ArtifactFormat) -> str:
    """Render one transcript artifact without publishing it."""
    match artifact_format:
        case ArtifactFormat.TEXT:
            return f"{document.text}\n"
        case ArtifactFormat.MARKDOWN:
            return _render_markdown(document)
        case ArtifactFormat.JSON:
            payload = TranscriptJsonPayload(
                source_path=document.source_path,
                backend=document.backend,
                duration_seconds=document.duration_seconds,
                timing_quality=document.timing_quality,
                segments=document.segments,
                text=document.text,
            )
            return f"{json.dumps(payload.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n"
        case ArtifactFormat.SRT:
            return _render_srt(document)
        case ArtifactFormat.VTT:
            return _render_vtt(document)


def _render_markdown(document: TranscriptDocument) -> str:
    lines = [
        f"# Transcript: {document.source_path.name}",
        "",
        "> Timing is coarse source-window timing, not word- or sentence-level timing.",
        "",
    ]
    for segment in document.segments:
        lines.extend(
            (
                "## "
                + _format_timestamp(segment.start_seconds, separator=".")
                + " - "
                + _format_timestamp(segment.end_seconds, separator="."),
                "",
                segment.text.strip(),
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_srt(document: TranscriptDocument) -> str:
    cues = [
        "\n".join(
            (
                str(segment.index + 1),
                _format_timestamp(segment.start_seconds, separator=",")
                + " --> "
                + _format_timestamp(segment.end_seconds, separator=","),
                segment.text.strip(),
            )
        )
        for segment in document.segments
    ]
    return "\n\n".join(cues) + "\n"


def _render_vtt(document: TranscriptDocument) -> str:
    cues = [
        "\n".join(
            (
                _format_timestamp(segment.start_seconds, separator=".")
                + " --> "
                + _format_timestamp(segment.end_seconds, separator="."),
                segment.text.strip(),
            )
        )
        for segment in document.segments
    ]
    header = "WEBVTT\n\nNOTE Timing is coarse source-window timing, not word timing."
    body = "\n\n".join(cues)
    return f"{header}\n\n{body}\n"


def _format_timestamp(seconds: float, *, separator: str) -> str:
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}{separator}{milliseconds:03}"
