"""Typed state values shared by the Granite worker supervisor."""

from dataclasses import dataclass
from uuid import UUID

from anyio.abc import ByteReceiveStream, Process


@dataclass(frozen=True, slots=True)
class PendingRequest:
    """Correlate one worker response with its request."""

    request_id: UUID
    chunk_index: int


@dataclass(frozen=True, slots=True)
class OpenWorker:
    """Own a newly opened worker and its output streams."""

    process: Process
    stdout: ByteReceiveStream
    stderr: ByteReceiveStream
