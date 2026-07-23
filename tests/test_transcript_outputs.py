from __future__ import annotations

from typing import TYPE_CHECKING

from stt_mcp.artifacts import publish_artifacts
from stt_mcp.backend import Backend
from stt_mcp.contracts import (
    ArtifactFormat,
    TimingQuality,
    TranscriptDocument,
    TranscriptJsonPayload,
    TranscriptSegment,
)
from stt_mcp.transcript import plan_source_windows, render_transcript

if TYPE_CHECKING:
    from pathlib import Path


def _document(source_path: Path) -> TranscriptDocument:
    return TranscriptDocument(
        source_path=source_path,
        backend=Backend.PARAKEET,
        duration_seconds=65.0,
        timing_quality=TimingQuality.COARSE_SOURCE_WINDOW,
        segments=(
            TranscriptSegment(index=0, start_seconds=0.0, end_seconds=30.0, text="First"),
            TranscriptSegment(index=1, start_seconds=30.0, end_seconds=60.0, text="Second"),
            TranscriptSegment(index=2, start_seconds=60.0, end_seconds=65.0, text="Third"),
        ),
    )


def test_plan_source_windows_clamps_final_window_to_media_duration() -> None:
    # Given
    duration_seconds = 65.0

    # When
    windows = plan_source_windows(duration_seconds=duration_seconds)

    # Then
    assert [(window.start_seconds, window.end_seconds) for window in windows] == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 65.0),
    ]


def test_json_output_declares_coarse_timing_quality(tmp_path: Path) -> None:
    # Given
    document = _document(tmp_path / "source.mp4")

    # When
    payload = TranscriptJsonPayload.model_validate_json(
        render_transcript(document, ArtifactFormat.JSON)
    )

    # Then
    assert payload.timing_quality is TimingQuality.COARSE_SOURCE_WINDOW
    assert payload.backend is Backend.PARAKEET
    assert payload.text == "First Second Third"


def test_srt_output_uses_monotonic_source_window_cues(tmp_path: Path) -> None:
    # Given
    document = _document(tmp_path / "source.mp4")

    # When
    rendered = render_transcript(document, ArtifactFormat.SRT)

    # Then
    assert "00:00:00,000 --> 00:00:30,000" in rendered
    assert "00:01:00,000 --> 00:01:05,000" in rendered


def test_vtt_output_contains_timing_disclaimer(tmp_path: Path) -> None:
    # Given
    document = _document(tmp_path / "source.mp4")

    # When
    rendered = render_transcript(document, ArtifactFormat.VTT)

    # Then
    assert rendered.startswith("WEBVTT\n\nNOTE Timing is coarse")


def test_publish_artifacts_atomically_replaces_files_without_staging_leaks(
    tmp_path: Path,
) -> None:
    # Given
    document = _document(tmp_path / "source.mp4")
    existing = tmp_path / "source.txt"
    _ = existing.write_text("old", encoding="utf-8")

    # When
    artifacts = publish_artifacts(
        document=document,
        formats=(ArtifactFormat.TEXT, ArtifactFormat.JSON),
        output_directory=tmp_path,
    )

    # Then
    assert existing.read_text(encoding="utf-8") == "First Second Third\n"
    assert {artifact.format for artifact in artifacts} == {
        ArtifactFormat.TEXT,
        ArtifactFormat.JSON,
    }
    assert tuple(tmp_path.glob("*.staging")) == ()
