"""Per-file atomic transcript artifact publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from stt_mcp.contracts import ArtifactFormat, ArtifactRecord, TranscriptDocument
from stt_mcp.transcript import render_transcript

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactPublishError(Exception):
    """Raised when a complete staged artifact cannot be published."""

    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"could not publish transcript artifact {self.path}: {self.detail}"


def publish_artifacts(
    *,
    document: TranscriptDocument,
    formats: tuple[ArtifactFormat, ...],
    output_directory: Path,
) -> tuple[ArtifactRecord, ...]:
    """Publish requested artifacts with atomic replacement per file."""
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[ArtifactRecord] = []
    seen: set[ArtifactFormat] = set()

    for artifact_format in formats:
        if artifact_format in seen:
            continue
        seen.add(artifact_format)
        destination = output_directory / f"{document.source_path.stem}.{artifact_format.value}"
        staging = destination.with_suffix(f"{destination.suffix}.staging")
        try:
            _ = staging.write_text(
                render_transcript(document, artifact_format),
                encoding="utf-8",
                newline="\n",
            )
            _ = staging.replace(destination)
        except OSError as error:
            raise ArtifactPublishError(path=destination, detail=str(error)) from error
        finally:
            staging.unlink(missing_ok=True)
        records.append(ArtifactRecord(format=artifact_format, path=destination))

    return tuple(records)
