"""Global CUDA memory preflight for safe model startup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, override

import pynvml

GIBIBYTE: Final = 1024**3
MINIMUM_FREE_BYTES: Final = 8 * GIBIBYTE
CUDA_DEVICE_INDEX: Final = 0


@dataclass(frozen=True, slots=True)
class CudaMemoryStatus:
    """NVML-reported global memory available on the CUDA device."""

    total_bytes: int
    free_bytes: int


@dataclass(frozen=True, slots=True)
class CudaCapacityError(Exception):
    """Raised when another workload leaves too little memory for model startup."""

    status: CudaMemoryStatus

    @override
    def __str__(self) -> str:
        available_gib = self.status.free_bytes / GIBIBYTE
        required_gib = MINIMUM_FREE_BYTES / GIBIBYTE
        return (
            f"CUDA device 0 has {available_gib:.1f} GiB free; STT-MCP enforces an "
            f"{required_gib:.1f} GiB startup safety floor. Pause other GPU workloads and try again."
        )


@dataclass(frozen=True, slots=True)
class CudaInspectionError(Exception):
    """Raised when NVML cannot inspect the required CUDA device."""

    detail: str

    @override
    def __str__(self) -> str:
        return f"could not inspect CUDA device 0 with NVML: {self.detail}"


@dataclass(frozen=True, slots=True)
class CudaDeviceSelectionError(Exception):
    """Raised when CUDA and NVML device identities cannot be matched safely."""

    device_count: int
    visible_devices: str | None

    @override
    def __str__(self) -> str:
        return (
            "STT-MCP currently requires exactly one NVIDIA GPU and no "
            f"CUDA_VISIBLE_DEVICES override; found device_count={self.device_count}, "
            f"CUDA_VISIBLE_DEVICES={self.visible_devices!r}"
        )


def require_unambiguous_cuda_device(*, device_count: int, visible_devices: str | None) -> None:
    """Require one physical GPU whose CUDA and NVML index are both zero."""
    if device_count != 1 or visible_devices not in (None, "0"):
        raise CudaDeviceSelectionError(
            device_count=device_count,
            visible_devices=visible_devices,
        )


def inspect_cuda_memory() -> CudaMemoryStatus:
    """Read global device memory through NVML rather than a process-local allocator."""
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        require_unambiguous_cuda_device(
            device_count=pynvml.nvmlDeviceGetCount(),
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        handle = pynvml.nvmlDeviceGetHandleByIndex(CUDA_DEVICE_INDEX)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return CudaMemoryStatus(total_bytes=memory.total, free_bytes=memory.free)
    except pynvml.NVMLError as error:
        raise CudaInspectionError(detail=str(error)) from error
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def require_cuda_capacity(status: CudaMemoryStatus) -> None:
    """Reject model startup when global free memory is below the observed safe floor."""
    if status.free_bytes < MINIMUM_FREE_BYTES:
        raise CudaCapacityError(status=status)


def ensure_cuda_capacity() -> None:
    """Inspect the CUDA device and require sufficient model-startup headroom."""
    require_cuda_capacity(inspect_cuda_memory())
