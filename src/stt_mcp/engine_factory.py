"""Lazy composition of the configured speech engine."""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING

from stt_mcp.backend import Backend, SpeechEngine
from stt_mcp.configuration import (
    ConfigNotFoundError,
    RuntimeConfig,
    load_runtime_config,
    runtime_config_path,
)

if TYPE_CHECKING:
    from pathlib import Path


def create_configured_engine(config_path: Path | None = None) -> SpeechEngine:
    """Create the one engine selected by persisted configuration."""
    path = config_path or runtime_config_path()
    try:
        config = load_runtime_config(path)
    except ConfigNotFoundError:
        config = RuntimeConfig(backend=Backend.GRANITE)
    return create_engine(config)


def create_engine(config: RuntimeConfig) -> SpeechEngine:
    """Load only the provider selected by configuration."""
    match config.backend:
        case Backend.GRANITE:
            from stt_mcp.runtime_policy import select_runtime
            from stt_mcp.supervisor import WorkerSupervisor

            runtime = select_runtime(platform=sys.platform, machine=platform.machine())
            return WorkerSupervisor(runtime)
        case Backend.PARAKEET:
            from stt_mcp.parakeet import ParakeetEngine

            return ParakeetEngine(settings=config.required_parakeet_settings())
