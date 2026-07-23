"""Cancellable parakeet.cpp speech engine adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, final, override

import anyio
from pydantic import BaseModel, ConfigDict, ValidationError

from stt_mcp.backend import AudioChunk, Backend, ParakeetDevice

if TYPE_CHECKING:
    from pathlib import Path

    from stt_mcp.configuration import ParakeetSettings


@dataclass(frozen=True, slots=True)
class ParakeetAssetNotFoundError(Exception):
    """Raised when the configured executable or model is absent."""

    path: Path

    @override
    def __str__(self) -> str:
        return f"Required Parakeet asset not found: {self.path}"


@dataclass(frozen=True, slots=True)
class ParakeetProcessError(Exception):
    """Raised when parakeet.cpp exits unsuccessfully."""

    returncode: int
    stderr: str

    @override
    def __str__(self) -> str:
        return f"parakeet.cpp exited with code {self.returncode}: {self.stderr}"


@dataclass(frozen=True, slots=True)
class ParakeetOutputError(Exception):
    """Raised when parakeet.cpp emits invalid JSON."""

    detail: str

    @override
    def __str__(self) -> str:
        return f"Invalid parakeet.cpp JSON output: {self.detail}"


class _ParakeetOutput(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    text: str


@final
class ParakeetEngine:
    """Speech engine backed by the pinned parakeet.cpp command-line interface."""

    _settings: ParakeetSettings
    _executable_arguments: tuple[str, ...]

    def __init__(
        self,
        settings: ParakeetSettings,
        *,
        executable_arguments: tuple[str, ...] = (),
    ) -> None:
        self._settings = settings
        self._executable_arguments = executable_arguments

    @property
    def backend(self) -> Backend:
        """Return Parakeet for transcript provenance."""
        return Backend.PARAKEET

    async def start(self) -> None:
        """Validate configured local assets."""
        for path in (self._settings.executable, self._settings.model):
            if not path.is_file():
                raise ParakeetAssetNotFoundError(path=path)

    async def transcribe(self, chunk: AudioChunk) -> str:
        """Run parakeet.cpp for one normalized WAV chunk."""
        await self.start()
        environment = dict(os.environ)
        environment["PARAKEET_DEVICE"] = _device_name(self._settings.device)
        completed = await anyio.run_process(
            [
                str(self._settings.executable),
                *self._executable_arguments,
                "transcribe",
                "--model",
                str(self._settings.model),
                "--input",
                str(chunk.path),
                "--decoder",
                "tdt",
                "--json",
            ],
            check=False,
            env=environment,
        )
        stderr = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        if completed.returncode != 0:
            raise ParakeetProcessError(
                returncode=completed.returncode,
                stderr=stderr,
            )
        try:
            output = _ParakeetOutput.model_validate_json(completed.stdout or b"")
        except ValidationError as error:
            raise ParakeetOutputError(detail=str(error)) from error
        return output.text.strip()

    async def aclose(self) -> None:
        """Complete the stateless engine lifecycle."""
        return


def _device_name(device: ParakeetDevice) -> str:
    match device:
        case ParakeetDevice.CPU:
            return "cpu"
        case ParakeetDevice.METAL:
            return "Metal0"
