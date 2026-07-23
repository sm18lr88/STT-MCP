"""Persisted speech backend configuration."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Self, override
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, model_validator

from stt_mcp.backend import Backend, ParakeetDevice

CONFIG_FILENAME: Final = "config.json"
CONFIG_PATH_ENVIRONMENT_VARIABLE: Final = "STT_MCP_CONFIG"


@dataclass(frozen=True, slots=True)
class ConfigNotFoundError(Exception):
    """Raised when no persisted runtime configuration exists."""

    path: Path

    @override
    def __str__(self) -> str:
        return f"STT-MCP configuration not found: {self.path}"


class ParakeetSettings(BaseModel):
    """Pinned local parakeet.cpp runtime paths and device."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    executable: Path
    model: Path
    device: ParakeetDevice


class RuntimeConfig(BaseModel):
    """Persisted concrete speech backend selection."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    backend: Backend
    parakeet: ParakeetSettings | None = None

    @model_validator(mode="after")
    def validate_backend_settings(self) -> Self:
        """Reject settings that do not match the selected backend."""
        match self.backend:
            case Backend.GRANITE:
                if self.parakeet is not None:
                    msg = "Granite configuration cannot include Parakeet settings"
                    raise ValueError(msg)
            case Backend.PARAKEET:
                if self.parakeet is None:
                    msg = "Parakeet configuration requires executable, model, and device"
                    raise ValueError(msg)
        return self

    def required_parakeet_settings(self) -> ParakeetSettings:
        """Return settings guaranteed by a validated Parakeet selection."""
        if self.parakeet is None:
            raise ConfigurationInvariantError(
                detail="validated Parakeet configuration has no Parakeet settings"
            )
        return self.parakeet


@dataclass(frozen=True, slots=True)
class ConfigurationInvariantError(Exception):
    """Raised when validated configuration violates an internal invariant."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


def default_config_path() -> Path:
    """Return the per-user STT-MCP configuration path."""
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "stt-mcp" / CONFIG_FILENAME


def runtime_config_path() -> Path:
    """Return the environment override or the default configuration path."""
    configured = os.environ.get(CONFIG_PATH_ENVIRONMENT_VARIABLE)
    return Path(configured) if configured is not None else default_config_path()


def load_runtime_config(path: Path) -> RuntimeConfig:
    """Parse a persisted runtime configuration."""
    if not path.is_file():
        raise ConfigNotFoundError(path=path)
    return RuntimeConfig.model_validate_json(path.read_bytes())


def save_runtime_config(config: RuntimeConfig, path: Path) -> None:
    """Atomically persist a runtime configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = path.with_name(f".{path.name}.{uuid4().hex}.staging")
    _ = staging_path.write_text(
        f"{config.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    _ = staging_path.replace(path)
