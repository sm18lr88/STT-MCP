from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from stt_mcp.worker_protocol import (
    MAX_LINE_BYTES,
    ProtocolLineTooLargeError,
    TranscribeCommand,
    WorkerProtocolError,
    decode_command,
    encode_message,
)


def test_transcribe_command_round_trips_through_ndjson() -> None:
    # Given
    command = TranscribeCommand(
        request_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        chunk_index=2,
        audio_path=Path("D:/audio/chunk.wav"),
    )

    # When
    decoded = decode_command(encode_message(command))

    # Then
    assert decoded == command


def test_decode_command_rejects_unknown_protocol_version() -> None:
    # Given
    line = b'{"protocol":"stt-worker","version":2,"type":"shutdown"}\n'

    # When / Then
    with pytest.raises(WorkerProtocolError, match="invalid worker command"):
        _ = decode_command(line)


def test_decode_command_rejects_unknown_fields() -> None:
    # Given
    line = b'{"protocol":"stt-worker","version":1,"type":"shutdown","extra":true}\n'

    # When / Then
    with pytest.raises(WorkerProtocolError, match="invalid worker command"):
        _ = decode_command(line)


def test_decode_command_rejects_oversized_lines() -> None:
    # Given
    line = b"x" * (MAX_LINE_BYTES + 1)

    # When / Then
    with pytest.raises(ProtocolLineTooLargeError):
        _ = decode_command(line)
