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
    def transcribe(self, *, input_features: torch.Tensor, attention_mask: torch.Tensor) -> _Output:
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


class _ShortAudioProcessor:
    def __call__(self, audios: list[torch.Tensor], *, device: str) -> GraniteInputs:
        _ = device
        waveform = audios[0]
        assert waveform.shape == (320,)
        assert torch.count_nonzero(waveform) == 0
        batched = waveform.unsqueeze(0)
        return GraniteInputs(input_features=batched, attention_mask=torch.ones_like(batched))

    def batch_decode(self, token_ids_list: list[torch.Tensor]) -> list[str]:
        assert len(token_ids_list) == 1
        return ["short"]


class _ShortAudioModel:
    def transcribe(self, *, input_features: torch.Tensor, attention_mask: torch.Tensor) -> _Output:
        assert input_features.shape == (1, 320)
        assert attention_mask.shape == (1, 320)
        return _Output(preds=[torch.tensor([1])])


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


def test_transcribe_pads_audio_shorter_than_feature_extractor_frame(tmp_path: Path) -> None:
    # Given
    audio_path = tmp_path / "short.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16_000)
        audio_file.writeframes(array("h", [0] * 100).tobytes())
    transcriber = GraniteTranscriber(
        model=_ShortAudioModel(),
        processor=_ShortAudioProcessor(),
        device="cpu",
    )

    # When
    text = transcriber.transcribe(audio_path)

    # Then
    assert text == "short"


def test_transcribe_preserves_minimum_feature_extractor_frame(tmp_path: Path) -> None:
    # Given
    audio_path = tmp_path / "minimum.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16_000)
        audio_file.writeframes(array("h", [0] * 320).tobytes())
    transcriber = GraniteTranscriber(
        model=_ShortAudioModel(),
        processor=_ShortAudioProcessor(),
        device="cpu",
    )

    # When / Then
    assert transcriber.transcribe(audio_path) == "short"
