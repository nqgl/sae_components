from functools import cached_property

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


class HuggingFaceModelConfig(ModelLoadingConfigBase[PreTrainedModel]):
    model_name: str = "gpt2"

    @property
    def name(self) -> str:  # type: ignore
        return self.model_name

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=DATA_DIRS.CACHE_DIR,
        )

    def _make_raw_model(
        self,
        load_as_dtype: torch.dtype | None = None,
        device: str | torch.device = "cuda",
    ):
        get_model_fn = AutoModelForCausalLM.from_pretrained
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
                torch_dtype=load_as_dtype,
                device_map=device,
            )
        return model

    def nnsight_wrap(self, model: PreTrainedModel) -> NNsight:
        nnsight_model = NNsight(model)
        if not hasattr(nnsight_model, "tokenizer"):
            setattr(nnsight_model, "tokenizer", self.tokenizer)
        return nnsight_model
