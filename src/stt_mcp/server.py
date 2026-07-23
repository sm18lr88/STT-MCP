"""FastMCP stdio server for local transcription."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession  # noqa: TC002 - FastMCP resolves annotations
from starlette.requests import Request  # noqa: TC002 - FastMCP resolves annotations

from stt_mcp.contracts import ArtifactFormat
from stt_mcp.engine_factory import create_configured_engine
from stt_mcp.service import TranscriptionResult, TranscriptionService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from stt_mcp.backend import SpeechEngine


@dataclass(frozen=True, slots=True)
class ServerContext:
    """Process-lifetime engine and application service."""

    engine: SpeechEngine
    service: TranscriptionService


@asynccontextmanager
async def _lifespan(_: FastMCP[ServerContext]) -> AsyncGenerator[ServerContext]:
    engine = create_configured_engine()
    try:
        yield ServerContext(engine=engine, service=TranscriptionService(engine))
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
    return await ctx.request_context.lifespan_context.service.transcribe(
        source_path=source,
        output_directory=destination,
        formats=requested,
    )


async def _resolve_path(raw_path: str) -> Path:
    expanded = await anyio.Path(raw_path).expanduser()
    return Path(await expanded.resolve())


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
