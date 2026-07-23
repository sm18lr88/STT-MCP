from __future__ import annotations

from typing import TYPE_CHECKING

from stt_mcp.backend import Backend, ParakeetDevice
from stt_mcp.configuration import (
    ParakeetSettings,
    RuntimeConfig,
    load_runtime_config,
    save_runtime_config,
)
from stt_mcp.hardware import HardwareProfile, recommend_backend

if TYPE_CHECKING:
    from pathlib import Path


def test_recommendation_selects_granite_for_one_safe_nvidia_gpu() -> None:
    # Given
    profile = HardwareProfile(
        platform="win32",
        machine="AMD64",
        nvidia_gpu_count=1,
        maximum_free_vram_bytes=9 * 1024**3,
    )

    # When
    recommendation = recommend_backend(profile)

    # Then
    assert recommendation.backend is Backend.GRANITE
    assert recommendation.parakeet_device is None


def test_recommendation_selects_metal_parakeet_for_apple_silicon() -> None:
    # Given
    profile = HardwareProfile(
        platform="darwin",
        machine="arm64",
        nvidia_gpu_count=0,
        maximum_free_vram_bytes=0,
    )

    # When
    recommendation = recommend_backend(profile)

    # Then
    assert recommendation.backend is Backend.PARAKEET
    assert recommendation.parakeet_device is ParakeetDevice.METAL


def test_recommendation_selects_cpu_parakeet_without_safe_cuda() -> None:
    # Given
    profile = HardwareProfile(
        platform="win32",
        machine="AMD64",
        nvidia_gpu_count=2,
        maximum_free_vram_bytes=24 * 1024**3,
    )

    # When
    recommendation = recommend_backend(profile)

    # Then
    assert recommendation.backend is Backend.PARAKEET
    assert recommendation.parakeet_device is ParakeetDevice.CPU


def test_runtime_config_round_trips_concrete_parakeet_choice(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "config.json"
    expected = RuntimeConfig(
        backend=Backend.PARAKEET,
        parakeet=ParakeetSettings(
            executable=tmp_path / "parakeet-cli.exe",
            model=tmp_path / "tdt-0.6b-v3-q4_k.gguf",
            device=ParakeetDevice.CPU,
        ),
    )

    # When
    save_runtime_config(expected, config_path)
    actual = load_runtime_config(config_path)

    # Then
    assert actual == expected


def test_granite_config_does_not_require_parakeet_assets(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "config.json"
    expected = RuntimeConfig(backend=Backend.GRANITE)

    # When
    save_runtime_config(expected, config_path)
    actual = load_runtime_config(config_path)

    # Then
    assert actual == expected
