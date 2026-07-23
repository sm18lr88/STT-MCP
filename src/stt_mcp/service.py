"""Shared transcription application service."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar, final

import anyio
from pydantic import BaseModel, ConfigDict

from stt_mcp.artifacts import publish_artifacts
from stt_mcp.backend import SpeechEngine, TranscriptionBusyError
from stt_mcp.contracts import (
    ArtifactFormat,
    ArtifactRecord,
    TimingQuality,
    TranscriptDocument,
    TranscriptSegment,
)
from stt_mcp.media import normalize_media


class TranscriptionResult(BaseModel):
    """Complete transcription response shared by CLI and MCP."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    document: TranscriptDocument
    artifacts: tuple[ArtifactRecord, ...]


@final
class TranscriptionService:
    """Orchestrate media normalization, inference, and artifact publication."""

    def __init__(self, engine: SpeechEngine) -> None:
        self._engine = engine
        self._request_lock = anyio.Lock()

    async def transcribe(
        self,
        *,
        source_path: Path,
        output_directory: Path,
        formats: tuple[ArtifactFormat, ...],
    ) -> TranscriptionResult:
        """Transcribe one media file and publish all requested artifacts."""
        if self._request_lock.locked():
            raise TranscriptionBusyError
        async with self._request_lock:
            return await self._transcribe_owned(
                source_path=source_path,
                output_directory=output_directory,
                formats=formats,
            )

    async def _transcribe_owned(
        self,
        *,
        source_path: Path,
        output_directory: Path,
        formats: tuple[ArtifactFormat, ...],
    ) -> TranscriptionResult:
        await self._engine.start()
        with TemporaryDirectory(prefix="stt-mcp-") as temporary_directory:
            chunks = await normalize_media(
                source_path=source_path,
                output_directory=Path(temporary_directory),
            )
            segment_list: list[TranscriptSegment] = []
            for chunk in chunks:
                segment_list.append(  # noqa: PERF401 - inference must remain sequential
                    TranscriptSegment(
                        index=chunk.index,
                        start_seconds=chunk.start_seconds,
                        end_seconds=chunk.end_seconds,
                        text=await self._engine.transcribe(chunk),
                    )
                )
            segments = tuple(segment_list)
        document = TranscriptDocument(
            source_path=source_path,
            backend=self._engine.backend,
            duration_seconds=chunks[-1].end_seconds,
            timing_quality=TimingQuality.COARSE_SOURCE_WINDOW,
            segments=segments,
        )
        artifacts = publish_artifacts(
            document=document,
            formats=formats,
            output_directory=output_directory,
        )
        return TranscriptionResult(document=document, artifacts=artifacts)
