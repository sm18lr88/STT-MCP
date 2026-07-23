"""FastMCP stdio server for local transcription."""

from __future__ import annotations

import platform
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from mcp.server.fastmcp import FastMCP

from stt_mcp.contracts import ArtifactFormat
from stt_mcp.runtime_policy import select_runtime
from stt_mcp.service import TranscriptionResult, TranscriptionService
from stt_mcp.supervisor import WorkerSupervisor

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_supervisor = WorkerSupervisor(select_runtime(platform=sys.platform, machine=platform.machine()))
_service = TranscriptionService(_supervisor)


@asynccontextmanager
async def _lifespan(_: FastMCP) -> AsyncGenerator[None]:
    yield
    await _supervisor.aclose()


mcp = FastMCP(
    "STT-MCP",
    instructions="Transcribe local audio or video files with IBM Granite Speech on CUDA.",
    lifespan=_lifespan,
)


@mcp.tool(structured_output=True)
async def transcribe(
    source_path: str,
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
    return await _service.transcribe(
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
