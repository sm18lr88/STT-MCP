"""CUDA runtime selection for supported host platforms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

if TYPE_CHECKING:
    from collections.abc import Mapping

SUPPORTED_MACHINES: Final = frozenset({"amd64", "x86_64"})


class RuntimePlatform(StrEnum):
    """Supported host operating systems."""

    WINDOWS = "win32"
    LINUX = "linux"


class AttentionImplementation(StrEnum):
    """Attention implementations selected for Granite inference."""

    SDPA = "sdpa"
    FLASH_ATTENTION_2 = "flash_attention_2"


PLATFORM_ATTENTION: Final[Mapping[RuntimePlatform, AttentionImplementation]] = MappingProxyType(
    {
        RuntimePlatform.WINDOWS: AttentionImplementation.SDPA,
        RuntimePlatform.LINUX: AttentionImplementation.FLASH_ATTENTION_2,
    }
)


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
    runtime_platform = _runtime_platform(platform=platform, machine=machine)
    if machine.casefold() not in SUPPORTED_MACHINES:
        raise UnsupportedRuntimeError(platform=platform, machine=machine)
    attention = PLATFORM_ATTENTION[runtime_platform]
    return RuntimePlan(
        platform=runtime_platform,
        attention=attention,
        requires_flash_attention=attention is AttentionImplementation.FLASH_ATTENTION_2,
    )


def _runtime_platform(*, platform: str, machine: str) -> RuntimePlatform:
    try:
        return RuntimePlatform(platform)
    except ValueError as error:
        raise UnsupportedRuntimeError(platform=platform, machine=machine) from error
