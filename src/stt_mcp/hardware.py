"""Hardware inspection and backend recommendation policy."""

from __future__ import annotations

import os
import platform as host_platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict

from stt_mcp.backend import Backend, ParakeetDevice

GRANITE_FREE_VRAM_FLOOR_BYTES: Final = 8 * 1024**3
MEBIBYTE_BYTES: Final = 1024**2
NVIDIA_SMI_TIMEOUT_SECONDS: Final = 10.0
WINDOWS_CREATE_NO_WINDOW: Final = 0x08000000


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Hardware facts used to recommend a speech backend."""

    platform: str
    machine: str
    nvidia_gpu_count: int
    maximum_free_vram_bytes: int


@unique
class RecommendationReason(StrEnum):
    """Stable reason codes consumed by setup agents."""

    SAFE_SINGLE_NVIDIA_GPU = "safe_single_nvidia_gpu"
    APPLE_SILICON = "apple_silicon"
    CPU_FALLBACK = "cpu_fallback"


@dataclass(frozen=True, slots=True)
class BackendRecommendation:
    """Concrete recommendation for a user to confirm during setup."""

    backend: Backend
    parakeet_device: ParakeetDevice | None
    reason: RecommendationReason


class HardwareInspection(BaseModel):
    """Machine-readable hardware recommendation returned to setup agents."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    platform: str
    machine: str
    nvidia_gpu_count: int
    maximum_free_vram_bytes: int
    recommended_backend: Backend
    parakeet_device: ParakeetDevice | None
    reason: RecommendationReason


def recommend_backend(profile: HardwareProfile) -> BackendRecommendation:
    """Recommend Granite only when its conservative startup policy is satisfied."""
    granite_safe = (
        profile.platform in {"win32", "linux"}
        and profile.machine.lower() in {"amd64", "x86_64"}
        and profile.nvidia_gpu_count == 1
        and profile.maximum_free_vram_bytes >= GRANITE_FREE_VRAM_FLOOR_BYTES
    )
    if granite_safe:
        return BackendRecommendation(
            backend=Backend.GRANITE,
            parakeet_device=None,
            reason=RecommendationReason.SAFE_SINGLE_NVIDIA_GPU,
        )

    apple_silicon = (
        profile.platform == "darwin"
        and profile.machine.lower() in {"arm64", "aarch64"}
    )
    if apple_silicon:
        return BackendRecommendation(
            backend=Backend.PARAKEET,
            parakeet_device=ParakeetDevice.METAL,
            reason=RecommendationReason.APPLE_SILICON,
        )

    return BackendRecommendation(
        backend=Backend.PARAKEET,
        parakeet_device=ParakeetDevice.CPU,
        reason=RecommendationReason.CPU_FALLBACK,
    )


def inspect_hardware() -> HardwareInspection:
    """Inspect the host without importing optional Granite dependencies."""
    free_vram_bytes = _nvidia_free_vram_bytes()
    profile = HardwareProfile(
        platform=sys.platform,
        machine=host_platform.machine(),
        nvidia_gpu_count=len(free_vram_bytes),
        maximum_free_vram_bytes=max(free_vram_bytes, default=0),
    )
    recommendation = recommend_backend(profile)
    return HardwareInspection(
        platform=profile.platform,
        machine=profile.machine,
        nvidia_gpu_count=profile.nvidia_gpu_count,
        maximum_free_vram_bytes=profile.maximum_free_vram_bytes,
        recommended_backend=recommendation.backend,
        parakeet_device=recommendation.parakeet_device,
        reason=recommendation.reason,
    )


def _nvidia_free_vram_bytes() -> tuple[int, ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    creation_flags = WINDOWS_CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(  # noqa: S603 - executable comes from shutil.which
            [
                executable,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if completed.returncode != 0:
        return ()
    free_vram_mib: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            free_vram_mib.append(int(line.strip()))
        except ValueError:
            continue
    return tuple(value * MEBIBYTE_BYTES for value in free_vram_mib)
