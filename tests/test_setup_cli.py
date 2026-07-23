from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from stt_mcp.backend import Backend, ParakeetDevice
from stt_mcp.cli import app
from stt_mcp.configuration import (
    ParakeetSettings,
    RuntimeConfig,
    load_runtime_config,
    save_runtime_config,
)
from stt_mcp.hardware import HardwareInspection

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_setup_inspect_emits_machine_readable_recommendation() -> None:
    # Given
    runner = CliRunner()

    # When
    result = runner.invoke(app, ["setup", "inspect"])

    # Then
    assert result.exit_code == 0
    inspection = HardwareInspection.model_validate_json(result.stdout)
    assert inspection.recommended_backend in {Backend.GRANITE, Backend.PARAKEET}


def test_setup_configure_persists_confirmed_parakeet_choice(tmp_path: Path) -> None:
    # Given
    runner = CliRunner()
    config_path = tmp_path / "config.json"
    executable_path = tmp_path / "parakeet-cli.exe"
    model_path = tmp_path / "model.gguf"

    # When
    result = runner.invoke(
        app,
        [
            "setup",
            "configure",
            "--backend",
            "parakeet",
            "--parakeet-executable",
            str(executable_path),
            "--parakeet-model",
            str(model_path),
            "--parakeet-device",
            "cpu",
            "--config-path",
            str(config_path),
        ],
    )

    # Then
    assert result.exit_code == 0
    config = load_runtime_config(config_path)
    assert config.backend is Backend.PARAKEET
    assert config.parakeet is not None
    assert config.parakeet.device is ParakeetDevice.CPU


def test_setup_configure_honors_environment_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    runner = CliRunner()
    config_path = tmp_path / "configured" / "config.json"
    fallback_root = tmp_path / "platform-default"
    monkeypatch.setenv("STT_MCP_CONFIG", str(config_path))
    monkeypatch.setenv("LOCALAPPDATA", str(fallback_root))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fallback_root))

    # When
    result = runner.invoke(app, ["setup", "configure", "--backend", "granite"])

    # Then
    assert result.exit_code == 0
    assert result.stdout.strip() == str(config_path)
    assert load_runtime_config(config_path).backend is Backend.GRANITE


def test_importing_cli_does_not_import_granite_runtime() -> None:
    # Given
    statement = (
        "import sys; import stt_mcp.cli; "
        "blocked={'stt_mcp.supervisor','stt_mcp.cuda_capacity','torch','transformers'}; "
        "raise SystemExit(1 if blocked.intersection(sys.modules) else 0)"
    )

    # When
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode == 0, completed.stderr


def test_importing_server_does_not_import_granite_runtime() -> None:
    # Given
    statement = (
        "import sys; import stt_mcp.server; "
        "blocked={'stt_mcp.supervisor','stt_mcp.cuda_capacity','torch','transformers'}; "
        "raise SystemExit(1 if blocked.intersection(sys.modules) else 0)"
    )

    # When
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode == 0, completed.stderr


def test_parakeet_factory_does_not_import_granite_runtime(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "config.json"
    save_runtime_config(
        RuntimeConfig(
            backend=Backend.PARAKEET,
            parakeet=ParakeetSettings(
                executable=tmp_path / "parakeet-cli.exe",
                model=tmp_path / "model.gguf",
                device=ParakeetDevice.CPU,
            ),
        ),
        config_path,
    )
    statement = (
        "import sys; from pathlib import Path; "
        "from stt_mcp.engine_factory import create_configured_engine; "
        f"engine=create_configured_engine(Path({str(config_path)!r})); "
        "blocked={'stt_mcp.supervisor','stt_mcp.cuda_capacity','torch','transformers'}; "
        "raise SystemExit(1 if blocked.intersection(sys.modules) else 0)"
    )

    # When
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode == 0, completed.stderr
