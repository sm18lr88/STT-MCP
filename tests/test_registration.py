from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from stt_mcp.registration import (
    JSON_OBJECT_ADAPTER,
    ClientConfigError,
    RegistrationClient,
    default_config_path,
    register_client,
    unregister_client,
)


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


def test_opencode_default_path_honors_xdg_config_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    # When
    path = default_config_path(RegistrationClient.OPENCODE)

    # Then
    assert path == tmp_path / "opencode" / "opencode.json"


@pytest.mark.parametrize("configured", ["", "relative/config"])
def test_opencode_default_path_ignores_invalid_xdg_base(
    configured: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", configured)

    path = default_config_path(RegistrationClient.OPENCODE)

    assert path == Path.home() / ".config" / "opencode" / "opencode.json"


def test_claude_default_path_rejects_empty_appdata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stt_mcp.registration.sys.platform", "win32")
    monkeypatch.setenv("APPDATA", "")

    with pytest.raises(ClientConfigError, match="unavailable or invalid"):
        _ = default_config_path(RegistrationClient.CLAUDE_DESKTOP)


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


@pytest.mark.skipif(os.name == "nt", reason="symlink creation is not portable on Windows")
def test_register_preserves_symlinked_configuration(tmp_path: Path) -> None:
    target_path = tmp_path / "managed" / "opencode.json"
    target_path.parent.mkdir()
    _ = target_path.write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "opencode.json"
    config_path.symlink_to(target_path)

    register_client(
        client=RegistrationClient.OPENCODE,
        config_path=config_path,
        executable=tmp_path / "python",
    )

    payload = JSON_OBJECT_ADAPTER.validate_json(target_path.read_bytes())
    assert config_path.is_symlink()
    assert payload["mcp"] == {
        "stt-mcp": {
            "type": "local",
            "command": [str(tmp_path / "python"), "-m", "stt_mcp.server"],
            "enabled": True,
        }
    }
