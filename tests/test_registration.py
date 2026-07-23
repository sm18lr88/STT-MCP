from __future__ import annotations

import json
import os
import stat
from typing import TYPE_CHECKING

import pytest

from stt_mcp.registration import (
    JSON_OBJECT_ADAPTER,
    RegistrationClient,
    register_client,
    unregister_client,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_register_opencode_preserves_existing_configuration(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "opencode.json"
    _ = config_path.write_text(
        json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "model": "example/model",
                "mcp": {"existing": {"type": "local", "command": ["existing"]}},
            }
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "python.exe"

    # When
    register_client(
        client=RegistrationClient.OPENCODE,
        config_path=config_path,
        executable=executable,
    )

    # Then
    payload = JSON_OBJECT_ADAPTER.validate_json(config_path.read_bytes())
    assert payload == {
        "$schema": "https://opencode.ai/config.json",
        "model": "example/model",
        "mcp": {
            "existing": {"type": "local", "command": ["existing"]},
            "stt-mcp": {
                "type": "local",
                "command": [str(executable), "-m", "stt_mcp.server"],
                "enabled": True,
            },
        },
    }


def test_register_claude_desktop_uses_stdio_module_command(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "claude_desktop_config.json"
    executable = tmp_path / "python.exe"

    # When
    register_client(
        client=RegistrationClient.CLAUDE_DESKTOP,
        config_path=config_path,
        executable=executable,
    )

    # Then
    payload = JSON_OBJECT_ADAPTER.validate_json(config_path.read_bytes())
    assert payload == {
        "mcpServers": {
            "stt-mcp": {
                "command": str(executable),
                "args": ["-m", "stt_mcp.server"],
            }
        }
    }


def test_unregister_removes_only_stt_mcp_entry(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "opencode.json"
    _ = config_path.write_text(
        json.dumps(
            {
                "mcp": {
                    "stt-mcp": {"type": "local", "command": ["python"]},
                    "existing": {"type": "local", "command": ["existing"]},
                }
            }
        ),
        encoding="utf-8",
    )

    # When
    removed = unregister_client(
        client=RegistrationClient.OPENCODE,
        config_path=config_path,
    )

    # Then
    payload = JSON_OBJECT_ADAPTER.validate_json(config_path.read_bytes())
    assert removed is True
    assert payload == {"mcp": {"existing": {"type": "local", "command": ["existing"]}}}


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not available on Windows")
def test_register_preserves_private_posix_mode(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "opencode.json"
    _ = config_path.write_text("{}\n", encoding="utf-8")
    config_path.chmod(0o600)

    # When
    register_client(
        client=RegistrationClient.OPENCODE,
        config_path=config_path,
        executable=tmp_path / "python",
    )

    # Then
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
