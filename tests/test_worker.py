from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING
from uuid import UUID

import pytest
import torch

from stt_mcp.model import MODEL_ID, MODEL_REVISION, GraniteTranscriber, ModelContractError
from stt_mcp.runtime_policy import AttentionImplementation
from stt_mcp.worker import run_worker
from stt_mcp.worker_protocol import (
    ErrorEvent,
    InitializeCommand,
    ModelSpec,
    ReadyEvent,
    ResultEvent,
    RuntimeSpec,
    ShutdownCommand,
    ShutdownCompleteEvent,
    TranscribeCommand,
    WorkerErrorCode,
    decode_event,
    encode_message,
)

if TYPE_CHECKING:
    from pathlib import Path

INITIALIZE_COMMAND = InitializeCommand(
    model=ModelSpec(model_id=MODEL_ID, revision=MODEL_REVISION),
    runtime=RuntimeSpec(device="cuda", attention=AttentionImplementation.SDPA),
)


def test_worker_emits_shutdown_confirmation() -> None:
    # Given
    input_stream = BytesIO(encode_message(ShutdownCommand()))
    output_stream = BytesIO()

    # When
    run_worker(input_stream, output_stream)

    # Then
    assert isinstance(decode_event(output_stream.getvalue()), ShutdownCompleteEvent)


def test_worker_rejects_transcription_before_initialization(tmp_path: Path) -> None:
    # Given
    command = TranscribeCommand(
        request_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        chunk_index=2,
        audio_path=tmp_path / "audio" / "chunk.wav",
    )
    input_stream = BytesIO(encode_message(command))
    output_stream = BytesIO()

    # When
    run_worker(input_stream, output_stream)

    # Then
    event = decode_event(output_stream.getvalue())
    assert isinstance(event, ErrorEvent)
    assert event.code is WorkerErrorCode.INITIALIZATION_FAILED
    assert event.worker_reusable is False


def test_worker_rejects_unpinned_model_identity() -> None:
    command = InitializeCommand(
        model=ModelSpec(model_id="untrusted/model", revision=MODEL_REVISION),
        runtime=RuntimeSpec(device="cuda", attention=AttentionImplementation.SDPA),
    )
    output_stream = BytesIO()

    run_worker(BytesIO(encode_message(command)), output_stream)

    event = decode_event(output_stream.getvalue())
    assert isinstance(event, ErrorEvent)
    assert event.code is WorkerErrorCode.INITIALIZATION_FAILED
    assert event.worker_reusable is False


def test_worker_reports_model_loading_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load(*, device: str, attention: AttentionImplementation) -> GraniteTranscriber:
        message = f"failed on {device} with {attention}"
        raise RuntimeError(message)

    monkeypatch.setattr(GraniteTranscriber, "load", staticmethod(fail_load))
    output_stream = BytesIO()

    run_worker(BytesIO(encode_message(INITIALIZE_COMMAND)), output_stream)

    event = decode_event(output_stream.getvalue())
    assert isinstance(event, ErrorEvent)
    assert event.code is WorkerErrorCode.INITIALIZATION_FAILED
    assert event.worker_reusable is False
    assert "failed on cuda" in event.message


def test_worker_transcribes_after_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber = object.__new__(GraniteTranscriber)

    def fake_load(*, device: str, attention: AttentionImplementation) -> GraniteTranscriber:
        assert device == "cuda"
        assert attention is AttentionImplementation.SDPA
        return transcriber

    def fake_transcribe(_transcriber: GraniteTranscriber, audio_path: Path) -> str:
        assert audio_path == tmp_path / "chunk.wav"
        return "hello from granite"

    monkeypatch.setattr(GraniteTranscriber, "load", staticmethod(fake_load))
    monkeypatch.setattr(GraniteTranscriber, "transcribe", fake_transcribe)
    command = TranscribeCommand(
        request_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        chunk_index=3,
        audio_path=tmp_path / "chunk.wav",
    )
    input_stream = BytesIO(
        encode_message(INITIALIZE_COMMAND)
        + encode_message(command)
        + encode_message(ShutdownCommand())
    )
    output_stream = BytesIO()

    run_worker(input_stream, output_stream)

    events = [decode_event(line) for line in output_stream.getvalue().splitlines(keepends=True)]
    assert len(events) == 3
    assert isinstance(events[0], ReadyEvent)
    assert isinstance(events[1], ResultEvent)
    assert events[1].request_id == command.request_id
    assert events[1].chunk_index == command.chunk_index
    assert events[1].text == "hello from granite"
    assert isinstance(events[2], ShutdownCompleteEvent)


@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_reusability"),
    [
        (torch.OutOfMemoryError("out of memory"), WorkerErrorCode.CUDA_OUT_OF_MEMORY, "terminal"),
        (ModelContractError(detail="invalid audio"), WorkerErrorCode.INVALID_AUDIO, "reusable"),
        (RuntimeError("inference failed"), WorkerErrorCode.INFERENCE_FAILED, "terminal"),
    ],
)
def test_worker_classifies_transcription_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: RuntimeError | ModelContractError,
    expected_code: WorkerErrorCode,
    expected_reusability: str,
) -> None:
    transcriber = object.__new__(GraniteTranscriber)

    def fake_load(*, device: str, attention: AttentionImplementation) -> GraniteTranscriber:
        assert device == "cuda"
        assert attention is AttentionImplementation.SDPA
        return transcriber

    def fail_transcribe(_transcriber: GraniteTranscriber, audio_path: Path) -> str:
        assert audio_path == command.audio_path
        raise failure

    monkeypatch.setattr(GraniteTranscriber, "load", staticmethod(fake_load))
    monkeypatch.setattr(GraniteTranscriber, "transcribe", fail_transcribe)
    command = TranscribeCommand(
        request_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        chunk_index=5,
        audio_path=tmp_path / "chunk.wav",
    )
    input_stream = BytesIO(
        encode_message(INITIALIZE_COMMAND)
        + encode_message(command)
        + encode_message(ShutdownCommand())
    )
    output_stream = BytesIO()

    run_worker(input_stream, output_stream)

    events = [decode_event(line) for line in output_stream.getvalue().splitlines(keepends=True)]
    assert isinstance(events[0], ReadyEvent)
    assert isinstance(events[1], ErrorEvent)
    assert events[1].code is expected_code
    assert events[1].worker_reusable is (expected_reusability == "reusable")
    assert len(events) == (3 if expected_reusability == "reusable" else 2)
