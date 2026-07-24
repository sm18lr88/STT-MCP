"""Failures raised while owning the persistent Granite worker."""

from typing import Final, override

import anyio

from stt_mcp.worker_protocol import WorkerProtocolError

WORKER_COMMUNICATION_EXCEPTIONS: Final[tuple[type[Exception], ...]] = (
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    ConnectionResetError,
    WorkerProtocolError,
)


class WorkerBusyError(Exception):
    """Raised when another transcription owns the worker."""

    @override
    def __str__(self) -> str:
        return "STT-MCP is busy with another transcription"


class WorkerFailureError(Exception):
    """Raised when the worker cannot satisfy a request."""

    detail: str

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    @override
    def __str__(self) -> str:
        return self.detail
