from typing import Self, TypedDict

from torch import Tensor, dtype

class PreTrainedModel:
    def eval(self) -> Self: ...

class _GraniteInputs(TypedDict):
    input_features: Tensor
    attention_mask: Tensor

class GraniteSpeechProcessor:
    def __call__(self, audios: list[Tensor], *, device: str) -> _GraniteInputs: ...
    def batch_decode(self, token_ids_list: list[Tensor]) -> list[str]: ...

class AutoModel:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        revision: str,
        trust_remote_code: bool,
        attn_implementation: str,
        device_map: str,
        dtype: dtype,
    ) -> PreTrainedModel: ...

class AutoProcessor:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        revision: str,
        trust_remote_code: bool,
    ) -> GraniteSpeechProcessor: ...
