from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anyio
import pytest

from stt_mcp.backend import AudioChunk, Backend, TranscriptionBusyError
from stt_mcp.contracts import ArtifactFormat
from stt_mcp.service import (
    ProgressReporter,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionService,
)

if TYPE_CHECKING:
    from pathlib import Path


class FakeEngine:
    @property
    def backend(self) -> Backend:
        return Backend.PARAKEET

    async def start(self) -> None:
        return

    async def transcribe(self, chunk: AudioChunk) -> str:
        _ = chunk
        return "unused"

    async def aclose(self) -> None:
        return


@pytest.mark.anyio
async def test_service_rejects_concurrent_request_without_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TranscriptionService(FakeEngine())
    entered = anyio.Event()
    release = anyio.Event()
    results: list[TranscriptionResult] = []
    expected = cast("TranscriptionResult", object())

    async def fake_transcribe_owned(
        _service: TranscriptionService,
        request: TranscriptionRequest,
        progress: ProgressReporter | None,
    ) -> TranscriptionResult:
        assert request.source_path == tmp_path / "speech.wav"
        assert request.output_directory == tmp_path
        assert request.formats == (ArtifactFormat.TEXT,)
        assert progress is None
        entered.set()
        await release.wait()
        return expected

    async def run_first_request() -> None:
        results.append(
            await service.transcribe(
                TranscriptionRequest(
                    source_path=tmp_path / "speech.wav",
                    output_directory=tmp_path,
                    formats=(ArtifactFormat.TEXT,),
                )
            )
        )

    monkeypatch.setattr(TranscriptionService, "_transcribe_owned", fake_transcribe_owned)

    async with anyio.create_task_group() as tasks:
        _ = tasks.start_soon(run_first_request)
        await entered.wait()
        with pytest.raises(TranscriptionBusyError):
            _ = await service.transcribe(
                TranscriptionRequest(
                    source_path=tmp_path / "speech.wav",
                    output_directory=tmp_path,
                    formats=(ArtifactFormat.TEXT,),
                )
            )
        release.set()

    assert results == [expected]
