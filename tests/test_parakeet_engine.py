from __future__ import annotations

import os
import sys
from pathlib import Path

import anyio
import pytest

from stt_mcp.backend import Backend, ParakeetDevice
from stt_mcp.configuration import ParakeetSettings
from stt_mcp.media import NormalizedChunk
from stt_mcp.parakeet import ParakeetEngine, ParakeetProcessError


def _write_fake_cli(path: Path, source: str) -> None:
    _ = path.write_text(source, encoding="utf-8")


def _settings(tmp_path: Path) -> ParakeetSettings:
    model_path = tmp_path / "model.gguf"
    model_path.touch()
    return ParakeetSettings(
        executable=Path(sys.executable),
        model=model_path,
        device=ParakeetDevice.CPU,
    )


def _chunk(tmp_path: Path) -> NormalizedChunk:
    audio_path = tmp_path / "chunk.wav"
    audio_path.touch()
    return NormalizedChunk(
        index=0,
        path=audio_path,
        start_seconds=0.0,
        end_seconds=1.0,
    )


def test_parakeet_engine_returns_text_from_real_cli_json(tmp_path: Path) -> None:
    # Given
    fake_cli = tmp_path / "fake_parakeet.py"
    script_lines = (
        "import json, os, sys",
        "assert sys.argv[1] == 'transcribe'",
        "assert os.environ['PARAKEET_DEVICE'] == 'cpu'",
        "print(json.dumps({'text': 'hello from parakeet'}))",
    )
    _write_fake_cli(fake_cli, "\n".join(script_lines))
    engine = ParakeetEngine(
        settings=_settings(tmp_path),
        executable_arguments=(str(fake_cli),),
    )

    # When
    transcript = anyio.run(engine.transcribe, _chunk(tmp_path))

    # Then
    assert transcript == "hello from parakeet"
    assert engine.backend is Backend.PARAKEET


def test_parakeet_engine_reports_exit_code_and_stderr(tmp_path: Path) -> None:
    # Given
    fake_cli = tmp_path / "failing_parakeet.py"
    _write_fake_cli(
        fake_cli,
        "import sys; print('model load failed', file=sys.stderr); raise SystemExit(7)",
    )
    engine = ParakeetEngine(
        settings=_settings(tmp_path),
        executable_arguments=(str(fake_cli),),
    )

    # When / Then
    with pytest.raises(ParakeetProcessError) as captured:
        _ = anyio.run(engine.transcribe, _chunk(tmp_path))
    assert captured.value.returncode == 7
    assert captured.value.stderr == "model load failed"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process liveness probe")
def test_parakeet_engine_terminates_process_when_cancelled(tmp_path: Path) -> None:
    # Given
    fake_cli = tmp_path / "sleeping_parakeet.py"
    pid_path = tmp_path / "pid.txt"
    _write_fake_cli(
        fake_cli,
        "\n".join(
            (
                "import os, pathlib, time",
                f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))",
                "time.sleep(60)",
            )
        ),
    )
    engine = ParakeetEngine(
        settings=_settings(tmp_path),
        executable_arguments=(str(fake_cli),),
    )

    async def cancel_transcription() -> None:
        with anyio.fail_after(1):
            _ = await engine.transcribe(_chunk(tmp_path))

    # When
    with pytest.raises(TimeoutError):
        anyio.run(cancel_transcription)

    # Then
    process_id = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(process_id, 0)
