from __future__ import annotations

import wave
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from stt_mcp.model import GraniteInputs, GraniteTranscriber

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Output:
    preds: list[torch.Tensor] | None


class _Model:
    def transcribe(
        self, *, input_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> _Output:
        assert input_features.shape == (1, 16_000)
        assert attention_mask.shape == (1, 16_000)
        return _Output(preds=[torch.tensor([1, 2])])


class _Processor:
    def __call__(self, audios: list[torch.Tensor], *, device: str) -> GraniteInputs:
        waveform = audios[0].unsqueeze(0)
        return GraniteInputs(input_features=waveform, attention_mask=torch.ones_like(waveform))

    def batch_decode(self, token_ids_list: list[torch.Tensor]) -> list[str]:
        assert len(token_ids_list) == 1
        return [" hello world "]


def test_transcribe_decodes_normalized_mono_waveform(tmp_path: Path) -> None:
    # Given
    audio_path = tmp_path / "normalized.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16_000)
        audio_file.writeframes(array("h", [0] * 16_000).tobytes())
    transcriber = GraniteTranscriber(model=_Model(), processor=_Processor(), device="cpu")

    # When
    text = transcriber.transcribe(audio_path)

    # Then
    assert text == "hello world"
