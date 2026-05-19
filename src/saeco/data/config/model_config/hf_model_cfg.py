from functools import cached_property
from typing import Any

import torch
from nnsight import NNsight
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from saeco.data.config.locations import DATA_DIRS
from saeco.data.config.model_config.model_type_cfg_base import ModelLoadingConfigBase
from saeco.data.piler.dict_piler import DictBatch

_GATED_MODEL_MARKERS = ("401", "403", "unauthorized", "gated", "accept", "login")


def _raise_gated_model_hint(model_name: str, err: Exception) -> None:
    msg = str(err).lower()
    if any(m in msg for m in _GATED_MODEL_MARKERS):
        raise RuntimeError(
            f"Failed to load '{model_name}' — this model appears to be gated.\n"
            "Accept the license on the HuggingFace model page and set the "
            "HF_TOKEN environment variable (or run `huggingface-cli login`)."
        ) from err


class HuggingFaceModelConfig(ModelLoadingConfigBase[PreTrainedModel]):
    model_name: str = "gpt2"

    @property
    def name(self) -> str:  # type: ignore
        return self.model_name

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        try:
            return AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=DATA_DIRS.CACHE_DIR,
            )
        except Exception as err:
            _raise_gated_model_hint(self.model_name, err)
            raise

    def _make_raw_model(
        self,
        load_as_dtype: torch.dtype | None = None,
        device: str | torch.device = "cuda",
    ):
        get_model_fn = AutoModelForCausalLM.from_pretrained
        try:
            if load_as_dtype is None:
                model = get_model_fn(
                    self.model_name,
                    cache_dir=DATA_DIRS.CACHE_DIR,
                    device_map=device,
                )
            else:
                model = get_model_fn(
                    self.model_name,
                    cache_dir=DATA_DIRS.CACHE_DIR,
                    dtype=load_as_dtype,
                    device_map=device,
                )
        except Exception as err:
            _raise_gated_model_hint(self.model_name, err)
            raise
        return model

    def nnsight_wrap(self, model: PreTrainedModel) -> NNsight:
        nnsight_model = NNsight(model)
        if not hasattr(nnsight_model, "tokenizer"):
            nnsight_model.tokenizer = self.tokenizer
        return nnsight_model

    def unpack_model_inputs(
        self, input_data: torch.Tensor | DictBatch, extra_kwargs: dict[str, Any]
    ) -> tuple[list[Any], dict[str, Any]]:
        return [input_data], extra_kwargs
