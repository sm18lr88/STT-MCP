"""STT-MCP command-line interface."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Annotated

import anyio
import typer
from rich.console import Console

from stt_mcp.contracts import ArtifactFormat
from stt_mcp.registration import (
    RegistrationClient,
    default_config_path,
)
from stt_mcp.registration import (
    register_client as register_mcp_client,
)
from stt_mcp.registration import (
    unregister_client as unregister_mcp_client,
)
from stt_mcp.runtime_policy import select_runtime
from stt_mcp.server import main as run_server
from stt_mcp.service import TranscriptionResult, TranscriptionService
from stt_mcp.supervisor import WorkerSupervisor

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def transcribe(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output_directory: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    formats: Annotated[list[ArtifactFormat] | None, typer.Option("--format", "-f")] = None,
) -> None:
    """Transcribe one local audio or video file."""
    destination = output_directory or source.parent
    requested = tuple(formats) if formats is not None else tuple(ArtifactFormat)
    result = anyio.run(_transcribe, source.resolve(), destination.resolve(), requested)
    console.print(result.document.text)
    for artifact in result.artifacts:
        console.print(f"{artifact.format.value}: {artifact.path}")


async def _transcribe(
    source: Path,
    output_directory: Path,
    formats: tuple[ArtifactFormat, ...],
) -> TranscriptionResult:
    runtime = select_runtime(platform=sys.platform, machine=platform.machine())
    supervisor = WorkerSupervisor(runtime)
    try:
        return await TranscriptionService(supervisor).transcribe(
            source_path=source,
            output_directory=output_directory,
            formats=formats,
        )
    finally:
        await supervisor.aclose()


@app.command("register")
def register_command(
    client: Annotated[RegistrationClient, typer.Argument()],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Register STT-MCP with a supported MCP client."""
    config_path = config.resolve() if config is not None else default_config_path(client)
    register_mcp_client(
        client=client,
        config_path=config_path,
        executable=Path(sys.executable).resolve(),
    )
    console.print(f"Registered STT-MCP with {client.value}: {config_path}")


@app.command("unregister")
def unregister_command(
    client: Annotated[RegistrationClient, typer.Argument()],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Remove STT-MCP's entry from a supported MCP client."""
    config_path = config.resolve() if config is not None else default_config_path(client)
    removed = unregister_mcp_client(client=client, config_path=config_path)
    outcome = "Removed" if removed else "No registration found for"
    console.print(f"{outcome} STT-MCP in {client.value}: {config_path}")


@app.command()
def serve() -> None:
    """Run STT-MCP over the standard MCP stdio transport."""
    run_server()


def main() -> None:
    """Run the command-line application."""
    app()
