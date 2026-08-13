"""Shared transcription application service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar, Protocol, final

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


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    """Canonical inputs for one transcription operation."""

    source_path: Path
    output_directory: Path
    formats: tuple[ArtifactFormat, ...]


class ProgressReporter(Protocol):
    """Receive durable transcription progress snapshots."""

    async def __call__(
        self,
        phase: str,
        completed_chunks: int,
        total_chunks: int | None,
        message: str,
        /,
    ) -> None:
        """Store one progress update."""
        ...


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
        request: TranscriptionRequest,
        progress: ProgressReporter | None = None,
    ) -> TranscriptionResult:
        """Transcribe one media file and publish all requested artifacts."""
        try:
            self._request_lock.acquire_nowait()
        except anyio.WouldBlock:
            raise TranscriptionBusyError from None
        try:
            return await self._transcribe_owned(
                request=request,
                progress=progress,
            )
        finally:
            self._request_lock.release()

    async def _transcribe_owned(
        self,
        request: TranscriptionRequest,
        progress: ProgressReporter | None,
    ) -> TranscriptionResult:
        if progress is not None:
            await progress("starting_engine", 0, None, "Starting speech engine")
        await self._engine.start()
        if progress is not None:
            await progress("normalizing", 0, None, "Normalizing source media")
        with TemporaryDirectory(prefix="stt-mcp-") as temporary_directory:
            chunks = await normalize_media(
                source_path=request.source_path,
                output_directory=Path(temporary_directory),
            )
            segment_list: list[TranscriptSegment] = []
            if progress is not None:
                await progress("transcribing", 0, len(chunks), "Transcribing audio chunks")
            for chunk in chunks:
                segment_list.append(
                    TranscriptSegment(
                        index=chunk.index,
                        start_seconds=chunk.start_seconds,
                        end_seconds=chunk.end_seconds,
                        text=await self._engine.transcribe(chunk),
                    )
                )
                if progress is not None:
                    await progress(
                        "transcribing",
                        chunk.index + 1,
                        len(chunks),
                        "Transcribing audio chunks",
                    )
            segments = tuple(segment_list)
        document = TranscriptDocument(
            source_path=request.source_path,
            backend=self._engine.backend,
            duration_seconds=chunks[-1].end_seconds,
            timing_quality=TimingQuality.COARSE_SOURCE_WINDOW,
            segments=segments,
        )
        if progress is not None:
            await progress(
                "publishing",
                len(chunks),
                len(chunks),
                "Publishing transcript artifacts",
            )
        artifacts = publish_artifacts(
            document=document,
            formats=request.formats,
            output_directory=request.output_directory,
        )
        return TranscriptionResult(document=document, artifacts=artifacts)
