"""CUDA runtime selection for supported host platforms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, override

SUPPORTED_MACHINES: Final = frozenset({"amd64", "x86_64"})


class RuntimePlatform(StrEnum):
    """Supported host operating systems."""

    WINDOWS = "win32"
    LINUX = "linux"


class AttentionImplementation(StrEnum):
    """Attention implementations selected for Granite inference."""

    SDPA = "sdpa"
    FLASH_ATTENTION_2 = "flash_attention_2"


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    """Effective CUDA runtime configuration."""

    platform: RuntimePlatform
    attention: AttentionImplementation
    requires_flash_attention: bool


@dataclass(frozen=True, slots=True)
class UnsupportedRuntimeError(Exception):
    """Raised when STT-MCP cannot run on the current host."""

    platform: str
    machine: str

    @override
    def __str__(self) -> str:
        return f"unsupported STT-MCP runtime: platform={self.platform}, machine={self.machine}"


def select_runtime(*, platform: str, machine: str) -> RuntimePlan:
    """Select the supported attention backend for a host."""
    try:
        runtime_platform = RuntimePlatform(platform)
    except ValueError as error:
        raise UnsupportedRuntimeError(platform=platform, machine=machine) from error

    if machine.casefold() not in SUPPORTED_MACHINES:
        raise UnsupportedRuntimeError(platform=platform, machine=machine)

    match runtime_platform:
        case RuntimePlatform.WINDOWS:
            return RuntimePlan(
                platform=runtime_platform,
                attention=AttentionImplementation.SDPA,
                requires_flash_attention=False,
            )
        case RuntimePlatform.LINUX:
            return RuntimePlan(
                platform=runtime_platform,
                attention=AttentionImplementation.FLASH_ATTENTION_2,
                requires_flash_attention=True,
            )
