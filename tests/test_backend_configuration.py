from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from stt_mcp.backend import Backend, ParakeetDevice
from stt_mcp.configuration import (
    ParakeetSettings,
    RuntimeConfig,
    default_config_path,
    load_runtime_config,
    save_runtime_config,
)
from stt_mcp.hardware import HardwareProfile, inspect_hardware, recommend_backend


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


def test_inspection_rejects_partial_nvidia_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_query(_executable: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="12288\nN/A\n",
            stderr="",
        )

    def fake_which(name: str) -> str:
        _ = name
        return "nvidia-smi"

    monkeypatch.setattr("stt_mcp.hardware.shutil.which", fake_which)
    monkeypatch.setattr("stt_mcp.hardware._query_nvidia_smi", fake_query)

    inspection = inspect_hardware()

    assert inspection.nvidia_gpu_count == 0
    assert inspection.maximum_free_vram_bytes == 0
    assert inspection.recommended_backend is Backend.PARAKEET


@pytest.mark.parametrize("configured", ["", "relative/config"])
def test_runtime_config_ignores_invalid_xdg_base(
    configured: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("stt_mcp.configuration.sys.platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", configured)

    path = default_config_path()

    assert path == Path.home() / ".config" / "stt-mcp" / "config.json"


def test_runtime_config_ignores_empty_windows_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stt_mcp.configuration.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "")

    path = default_config_path()

    assert path == Path.home() / "AppData" / "Local" / "stt-mcp" / "config.json"


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


def test_granite_config_rejects_parakeet_assets(tmp_path: Path) -> None:
    # Given
    parakeet = ParakeetSettings(
        executable=tmp_path / "parakeet-cli.exe",
        model=tmp_path / "model.gguf",
        device=ParakeetDevice.CPU,
    )

    # When / Then
    with pytest.raises(ValidationError, match="cannot include Parakeet settings"):
        _ = RuntimeConfig(backend=Backend.GRANITE, parakeet=parakeet)


def test_parakeet_config_requires_runtime_assets() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError, match="requires executable, model, and device"):
        _ = RuntimeConfig(backend=Backend.PARAKEET)
