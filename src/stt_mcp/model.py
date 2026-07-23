"""Pinned IBM Granite Speech model loading and inference."""

from __future__ import annotations

import sys
import wave
from array import array
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Protocol,
    TypedDict,
    final,
    override,
    runtime_checkable,
)

import torch
from transformers import AutoModel, AutoProcessor

if TYPE_CHECKING:
    from pathlib import Path

    from stt_mcp.runtime_policy import AttentionImplementation

MODEL_ID: Final = "ibm-granite/granite-speech-4.1-2b-nar"
MODEL_REVISION: Final = "99a4df9007ac5682f9daa093fb7008ff606e9a5d"
EXPECTED_SAMPLE_RATE: Final = 16_000
PCM16_SAMPLE_WIDTH: Final = 2


class GraniteInputs(TypedDict):
    """Tensor inputs produced by the pinned Granite processor."""

    input_features: torch.Tensor
    attention_mask: torch.Tensor


class GraniteOutput(Protocol):
    """Inference output exposed by the pinned remote model code."""

    @property
    def preds(self) -> list[torch.Tensor] | None:
        """Return decoded token tensors when inference succeeds."""
        ...


@runtime_checkable
class GraniteModel(Protocol):
    """Runtime-checked custom model capability."""

    def transcribe(
        self, *, input_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> GraniteOutput:
        """Run non-autoregressive transcription."""
        ...


class GraniteProcessor(Protocol):
    """Runtime-checked custom processor capability."""

    def __call__(self, audios: list[torch.Tensor], *, device: str) -> GraniteInputs:
        """Extract model-ready audio features."""
        ...

    def batch_decode(self, token_ids_list: list[torch.Tensor]) -> list[str]:
        """Decode model token predictions."""
        ...


@dataclass(frozen=True, slots=True)
class ModelContractError(Exception):
    """Raised when the pinned model or audio violates its expected contract."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@final
class GraniteTranscriber:
    """Stateful owner of one loaded Granite model and processor."""

    __slots__: ClassVar[tuple[str, ...]] = ("_device", "_model", "_processor")

    _model: GraniteModel
    _processor: GraniteProcessor
    _device: str

    def __init__(
        self,
        *,
        model: GraniteModel,
        processor: GraniteProcessor,
        device: str,
    ) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    @classmethod
    def load(
        cls, *, device: str, attention: AttentionImplementation
    ) -> GraniteTranscriber:
        """Load the exact reviewed model revision on the selected CUDA backend."""
        loaded_model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            attn_implementation=attention.value,
            device_map=device,
            dtype=torch.bfloat16,
        ).eval()
        loaded_processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        if not isinstance(loaded_model, GraniteModel):
            raise ModelContractError(detail="pinned Granite model has no transcribe capability")
        return cls(model=loaded_model, processor=loaded_processor, device=device)

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe one normalized 16 kHz mono WAV file."""
        waveform = _load_pcm16_mono(audio_path)
        inputs = self._processor([waveform], device=self._device)
        with torch.inference_mode():
            output = self._model.transcribe(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
            )
        if output.preds is None:
            raise ModelContractError(detail="Granite model returned no predictions")
        decoded = self._processor.batch_decode(output.preds)
        if len(decoded) != 1:
            raise ModelContractError(
                detail=f"Granite model returned {len(decoded)} transcripts for one chunk"
            )
        return decoded[0].strip()


def _load_pcm16_mono(audio_path: Path) -> torch.Tensor:
    try:
        with wave.open(str(audio_path), "rb") as audio_file:
            if (
                audio_file.getframerate() != EXPECTED_SAMPLE_RATE
                or audio_file.getnchannels() != 1
                or audio_file.getsampwidth() != PCM16_SAMPLE_WIDTH
            ):
                raise ModelContractError(
                    detail=f"expected 16 kHz mono PCM16 WAV input: {audio_path}"
                )
            frames = audio_file.readframes(audio_file.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise ModelContractError(detail=f"could not read normalized WAV: {audio_path}") from error

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    return torch.tensor(samples, dtype=torch.float32) / 32_768.0
