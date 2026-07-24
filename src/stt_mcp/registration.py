"""Reversible registration with JSON-configured MCP clients."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

from filelock import FileLock, Timeout
from pydantic import TypeAdapter, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

SERVER_NAME: Final = "stt-mcp"
LOCK_TIMEOUT_SECONDS: Final = 30.0
type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
JSON_OBJECT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(JsonObject)


class RegistrationClient(StrEnum):
    """MCP clients with supported automated registration."""

    OPENCODE = "opencode"
    CLAUDE_DESKTOP = "claude-desktop"


CLIENT_SECTION_NAMES: Final[Mapping[RegistrationClient, str]] = MappingProxyType(
    {
        RegistrationClient.OPENCODE: "mcp",
        RegistrationClient.CLAUDE_DESKTOP: "mcpServers",
    }
)


@dataclass(frozen=True, slots=True)
class ClientConfigError(Exception):
    """Raised when an MCP client configuration cannot be preserved safely."""

    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"could not update {self.path}: {self.detail}"


def default_config_path(client: RegistrationClient) -> Path:
    """Return the conventional user configuration path for a supported client."""
    return CONFIG_PATH_RESOLVERS[client]()


def _opencode_config_path() -> Path:
    root = _absolute_environment_path("XDG_CONFIG_HOME") or Path.home() / ".config"
    return root / "opencode" / "opencode.json"


def _claude_desktop_config_path() -> Path:
    if sys.platform == "win32":
        return _windows_claude_desktop_config_path()
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def _windows_claude_desktop_config_path() -> Path:
    appdata = _absolute_environment_path("APPDATA")
    if appdata is None:
        raise ClientConfigError(
            path=Path.home(),
            detail="APPDATA is unavailable or invalid; pass --config explicitly",
        )
    return appdata / "Claude" / "claude_desktop_config.json"


def _absolute_environment_path(name: str) -> Path | None:
    configured = os.environ.get(name)
    if not configured:
        return None
    candidate = Path(configured)
    return candidate if candidate.is_absolute() else None


CONFIG_PATH_RESOLVERS: Final[Mapping[RegistrationClient, Callable[[], Path]]] = MappingProxyType(
    {
        RegistrationClient.OPENCODE: _opencode_config_path,
        RegistrationClient.CLAUDE_DESKTOP: _claude_desktop_config_path,
    }
)


def _opencode_server_entry(executable: Path) -> JsonObject:
    return {
        "type": "local",
        "command": [str(executable), "-m", "stt_mcp.server"],
        "enabled": True,
    }


def _claude_desktop_server_entry(executable: Path) -> JsonObject:
    return {
        "command": str(executable),
        "args": ["-m", "stt_mcp.server"],
    }


SERVER_ENTRY_BUILDERS: Final[Mapping[RegistrationClient, Callable[[Path], JsonObject]]] = (
    MappingProxyType(
        {
            RegistrationClient.OPENCODE: _opencode_server_entry,
            RegistrationClient.CLAUDE_DESKTOP: _claude_desktop_server_entry,
        }
    )
)


def register_client(
    *,
    client: RegistrationClient,
    config_path: Path,
    executable: Path,
) -> None:
    """Create or replace only STT-MCP's entry in a client configuration."""
    target_path = _config_target(config_path)
    with _locked_config(target_path):
        document = _read_config(target_path)
        section_name = CLIENT_SECTION_NAMES[client]
        servers = _read_section(document, section_name, target_path)
        servers[SERVER_NAME] = SERVER_ENTRY_BUILDERS[client](executable)
        document[section_name] = servers
        _publish_config(target_path, document)


def unregister_client(*, client: RegistrationClient, config_path: Path) -> bool:
    """Remove only STT-MCP's entry and report whether one existed."""
    target_path = _config_target(config_path)
    with _locked_config(target_path):
        if not target_path.exists():
            return False
        document = _read_config(target_path)
        section_name = CLIENT_SECTION_NAMES[client]
        servers = _read_section(document, section_name, target_path)
        if servers.pop(SERVER_NAME, None) is None:
            return False
        document[section_name] = servers
        _publish_config(target_path, document)
        return True


def _config_target(path: Path) -> Path:
    if not path.is_symlink():
        return path
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ClientConfigError(path=path, detail=str(error)) from error


def _read_config(path: Path) -> JsonObject:
    if not path.exists():
        return {}
    try:
        return JSON_OBJECT_ADAPTER.validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ClientConfigError(path=path, detail=str(error)) from error


def _read_section(document: JsonObject, name: str, path: Path) -> JsonObject:
    value = document.get(name)
    if value is None:
        return {}
    try:
        return JSON_OBJECT_ADAPTER.validate_python(value)
    except ValidationError as error:
        raise ClientConfigError(path=path, detail=f"{name} must be a JSON object") from error


def _publish_config(path: Path, document: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    staging_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".staging",
            dir=path.parent,
            delete=False,
        ) as staging:
            staging_path = Path(staging.name)
            _ = staging.write(f"{json.dumps(document, ensure_ascii=False, indent=2)}\n")
        staging_path.chmod(mode)
        _ = staging_path.replace(path)
    except OSError as error:
        raise ClientConfigError(path=path, detail=str(error)) from error
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)


@contextmanager
def _locked_config(path: Path) -> Generator[None]:
    digest = hashlib.sha256(os.fsencode(str(path.resolve()))).hexdigest()
    lock_directory = Path(gettempdir()) / "stt-mcp-locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_directory / f"{digest}.lock", timeout=LOCK_TIMEOUT_SECONDS)
    try:
        with lock:
            yield
    except Timeout as error:
        raise ClientConfigError(
            path=path,
            detail="timed out waiting for configuration lock",
        ) from error
