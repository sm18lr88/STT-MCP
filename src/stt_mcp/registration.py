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
from typing import TYPE_CHECKING, Final, override

from filelock import FileLock, Timeout
from pydantic import TypeAdapter, ValidationError

if TYPE_CHECKING:
    from collections.abc import Generator

SERVER_NAME: Final = "stt-mcp"
LOCK_TIMEOUT_SECONDS: Final = 30.0
type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
JSON_OBJECT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(JsonObject)


class RegistrationClient(StrEnum):
    """MCP clients with supported automated registration."""

    OPENCODE = "opencode"
    CLAUDE_DESKTOP = "claude-desktop"


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
    match client:
        case RegistrationClient.OPENCODE:
            config_home = os.environ.get("XDG_CONFIG_HOME")
            root = Path(config_home) if config_home is not None else Path.home() / ".config"
            return root / "opencode" / "opencode.json"
        case RegistrationClient.CLAUDE_DESKTOP:
            if sys.platform == "win32":
                appdata = os.environ.get("APPDATA")
                if appdata is None:
                    raise ClientConfigError(
                        path=Path.home(),
                        detail="APPDATA is unavailable; pass --config explicitly",
                    )
                return Path(appdata) / "Claude" / "claude_desktop_config.json"
            if sys.platform == "darwin":
                return (
                    Path.home()
                    / "Library"
                    / "Application Support"
                    / "Claude"
                    / "claude_desktop_config.json"
                )
            return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def register_client(
    *,
    client: RegistrationClient,
    config_path: Path,
    executable: Path,
) -> None:
    """Create or replace only STT-MCP's entry in a client configuration."""
    with _locked_config(config_path):
        document = _read_config(config_path)
        match client:
            case RegistrationClient.OPENCODE:
                servers = _read_section(document, "mcp", config_path)
                servers[SERVER_NAME] = {
                    "type": "local",
                    "command": [str(executable), "-m", "stt_mcp.server"],
                    "enabled": True,
                }
                document["mcp"] = servers
            case RegistrationClient.CLAUDE_DESKTOP:
                servers = _read_section(document, "mcpServers", config_path)
                servers[SERVER_NAME] = {
                    "command": str(executable),
                    "args": ["-m", "stt_mcp.server"],
                }
                document["mcpServers"] = servers
        _publish_config(config_path, document)


def unregister_client(*, client: RegistrationClient, config_path: Path) -> bool:
    """Remove only STT-MCP's entry and report whether one existed."""
    with _locked_config(config_path):
        if not config_path.exists():
            return False
        document = _read_config(config_path)
        match client:
            case RegistrationClient.OPENCODE:
                section_name = "mcp"
            case RegistrationClient.CLAUDE_DESKTOP:
                section_name = "mcpServers"
        servers = _read_section(document, section_name, config_path)
        if servers.pop(SERVER_NAME, None) is None:
            return False
        document[section_name] = servers
        _publish_config(config_path, document)
        return True


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
