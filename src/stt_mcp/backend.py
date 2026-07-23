"""Backend-neutral speech engine contracts."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path


class TranscriptionBusyError(RuntimeError):
    """Raised when another transcription owns the service."""


@unique
class Backend(StrEnum):
    """Available speech recognition engines."""

    GRANITE = "granite"
    PARAKEET = "parakeet"


@unique
class ParakeetDevice(StrEnum):
    """Execution devices supported by packaged parakeet.cpp releases."""

    CPU = "cpu"
    METAL = "metal"


class AudioChunk(Protocol):
    """Normalized audio window consumed by speech engines."""

    @property
    def index(self) -> int:
        """Return the zero-based source-window index."""
        ...

    @property
    def path(self) -> Path:
        """Return the normalized WAV path."""
        ...


class SpeechEngine(Protocol):
    """Backend-neutral speech recognition contract."""

    @property
    def backend(self) -> Backend:
        """Return the engine identity used for transcript provenance."""
        ...

    async def start(self) -> None:
        """Validate and initialize the selected engine."""
        ...

    async def transcribe(self, chunk: AudioChunk) -> str:
        """Transcribe one normalized WAV chunk."""
        ...

    async def aclose(self) -> None:
        """Release engine resources."""
        ...
