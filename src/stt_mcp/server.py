"""FastMCP stdio server for local transcription."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID  # noqa: TC003 - FastMCP resolves tool annotations at runtime

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession  # noqa: TC002 - FastMCP resolves annotations
from starlette.requests import Request  # noqa: TC002 - FastMCP resolves annotations

from stt_mcp.contracts import ArtifactFormat
from stt_mcp.engine_factory import create_configured_engine
from stt_mcp.job_contracts import (  # noqa: TC001 - FastMCP resolves tool annotations at runtime
    TranscriptionJobStatus,
)
from stt_mcp.jobs import TranscriptionJobManager
from stt_mcp.service import (
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionService,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from stt_mcp.backend import SpeechEngine


@dataclass(frozen=True, slots=True)
class ServerContext:
    """Process-lifetime engine and application service."""

    engine: SpeechEngine
    service: TranscriptionService
    jobs: TranscriptionJobManager


@asynccontextmanager
async def _lifespan(_: FastMCP[ServerContext]) -> AsyncGenerator[ServerContext]:
    engine = create_configured_engine()
    service = TranscriptionService(engine)
    async with anyio.create_task_group() as task_group:
        jobs = TranscriptionJobManager(service, task_group)
        try:
            yield ServerContext(engine=engine, service=service, jobs=jobs)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await jobs.aclose()
                finally:
                    await engine.aclose()


mcp = FastMCP[ServerContext](
    "STT-MCP",
    instructions="Transcribe local audio or video files with the configured speech backend.",
    lifespan=_lifespan,
)


@mcp.tool(structured_output=True)
async def transcribe(
    source_path: str,
    ctx: Context[ServerSession, ServerContext, Request],
    output_directory: str | None = None,
    formats: list[ArtifactFormat] | None = None,
) -> TranscriptionResult:
    """Transcribe local media and return the transcript plus published artifact paths."""
    source = await _resolve_path(source_path)
    destination = (
        await _resolve_path(output_directory)
        if output_directory is not None
        else source.parent
    )
    requested = tuple(formats) if formats is not None else tuple(ArtifactFormat)
    return await ctx.request_context.lifespan_context.jobs.transcribe(
        TranscriptionRequest(
            source_path=source,
            output_directory=destination,
            formats=requested,
        )
    )


@mcp.tool(structured_output=True)
async def start_transcription(
    job_id: UUID,
    source_path: str,
    ctx: Context[ServerSession, ServerContext, Request],
    output_directory: str | None = None,
    formats: list[ArtifactFormat] | None = None,
) -> TranscriptionJobStatus:
    """Start a process-lifetime transcription job and return immediately."""
    source = await _resolve_path(source_path)
    destination = (
        await _resolve_path(output_directory)
        if output_directory is not None
        else source.parent
    )
    requested = tuple(formats) if formats is not None else tuple(ArtifactFormat)
    return ctx.request_context.lifespan_context.jobs.start(
        job_id,
        TranscriptionRequest(
            source_path=source,
            output_directory=destination,
            formats=requested,
        ),
    )


@mcp.tool(structured_output=True)
async def get_transcription_status(
    job_id: UUID,
    ctx: Context[ServerSession, ServerContext, Request],
) -> TranscriptionJobStatus:
    """Return current progress, result, or terminal error for a job."""
    return ctx.request_context.lifespan_context.jobs.status(job_id)


@mcp.tool(structured_output=True)
async def cancel_transcription(
    job_id: UUID,
    ctx: Context[ServerSession, ServerContext, Request],
) -> TranscriptionJobStatus:
    """Request cancellation and return without waiting for cleanup."""
    return ctx.request_context.lifespan_context.jobs.cancel(job_id)


async def _resolve_path(raw_path: str) -> Path:
    expanded = await anyio.Path(raw_path).expanduser()
    return Path(await expanded.resolve())


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
