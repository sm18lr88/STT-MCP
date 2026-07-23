"""AI-guided backend setup commands."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Typer resolves annotations at runtime
from typing import Annotated

import typer

from stt_mcp.backend import Backend, ParakeetDevice
from stt_mcp.configuration import (
    ParakeetSettings,
    RuntimeConfig,
    runtime_config_path,
    save_runtime_config,
)
from stt_mcp.hardware import inspect_hardware

setup_app = typer.Typer(no_args_is_help=True)


@setup_app.command("inspect")
def inspect_command() -> None:
    """Emit hardware facts and a backend recommendation as JSON."""
    typer.echo(inspect_hardware().model_dump_json(indent=2))


@setup_app.command("configure")
def configure_command(
    backend: Annotated[Backend, typer.Option("--backend")],
    parakeet_executable: Annotated[Path | None, typer.Option()] = None,
    parakeet_model: Annotated[Path | None, typer.Option()] = None,
    parakeet_device: Annotated[ParakeetDevice | None, typer.Option()] = None,
    config_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Persist the concrete backend choice confirmed by the user."""
    match backend:
        case Backend.GRANITE:
            config = RuntimeConfig(backend=backend)
        case Backend.PARAKEET:
            if (
                parakeet_executable is None
                or parakeet_model is None
                or parakeet_device is None
            ):
                message = (
                    "Parakeet requires --parakeet-executable, --parakeet-model, "
                    "and --parakeet-device"
                )
                raise typer.BadParameter(message)
            config = RuntimeConfig(
                backend=backend,
                parakeet=ParakeetSettings(
                    executable=parakeet_executable,
                    model=parakeet_model,
                    device=parakeet_device,
                ),
            )
    destination = config_path or runtime_config_path()
    save_runtime_config(config, destination)
    typer.echo(str(destination))
