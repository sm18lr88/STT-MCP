from __future__ import annotations

import pytest

from stt_mcp.runtime_policy import (
    AttentionImplementation,
    RuntimePlatform,
    UnsupportedRuntimeError,
    select_runtime,
)


def test_windows_selects_cuda_sdpa_when_host_is_amd64() -> None:
    # Given
    platform = "win32"
    machine = "AMD64"

    # When
    plan = select_runtime(platform=platform, machine=machine)

    # Then
    assert plan.platform is RuntimePlatform.WINDOWS
    assert plan.attention is AttentionImplementation.SDPA
    assert plan.requires_flash_attention is False


def test_linux_selects_flash_attention_when_host_is_x86_64() -> None:
    # Given
    platform = "linux"
    machine = "x86_64"

    # When
    plan = select_runtime(platform=platform, machine=machine)

    # Then
    assert plan.platform is RuntimePlatform.LINUX
    assert plan.attention is AttentionImplementation.FLASH_ATTENTION_2
    assert plan.requires_flash_attention is True


def test_runtime_selection_rejects_unsupported_platform() -> None:
    # Given
    platform = "darwin"

    # When / Then
    with pytest.raises(UnsupportedRuntimeError, match="darwin"):
        _ = select_runtime(platform=platform, machine="arm64")


def test_runtime_selection_rejects_unsupported_machine() -> None:
    # Given
    machine = "arm64"

    # When / Then
    with pytest.raises(UnsupportedRuntimeError, match="arm64"):
        _ = select_runtime(platform="linux", machine=machine)
