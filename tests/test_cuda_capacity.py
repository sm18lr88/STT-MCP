from __future__ import annotations

import pytest

from stt_mcp import cuda_capacity


def test_capacity_rejects_memory_below_model_startup_floor() -> None:
    # Given
    status = cuda_capacity.CudaMemoryStatus(
        total_bytes=16 * 1024**3,
        free_bytes=cuda_capacity.MINIMUM_FREE_BYTES - 1,
    )

    # When / Then
    with pytest.raises(cuda_capacity.CudaCapacityError, match=r"8\.0 GiB"):
        cuda_capacity.require_cuda_capacity(status)


def test_capacity_accepts_memory_at_model_startup_floor() -> None:
    # Given
    status = cuda_capacity.CudaMemoryStatus(
        total_bytes=16 * 1024**3,
        free_bytes=cuda_capacity.MINIMUM_FREE_BYTES,
    )

    # When
    cuda_capacity.require_cuda_capacity(status)

    # Then


def test_device_selection_rejects_multiple_physical_gpus() -> None:
    # Given / When / Then
    with pytest.raises(cuda_capacity.CudaDeviceSelectionError, match="exactly one NVIDIA GPU"):
        cuda_capacity.require_unambiguous_cuda_device(device_count=2, visible_devices=None)


def test_device_selection_rejects_cuda_visibility_override() -> None:
    # Given / When / Then
    with pytest.raises(cuda_capacity.CudaDeviceSelectionError, match="CUDA_VISIBLE_DEVICES"):
        cuda_capacity.require_unambiguous_cuda_device(device_count=1, visible_devices="1")
