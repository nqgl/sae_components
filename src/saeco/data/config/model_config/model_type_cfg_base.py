from functools import cached_property

from nnsight import NNsight
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
)

from saeco.data.config.locations import DATA_DIRS

from saeco.sweeps import SweepableConfig
from typing import Protocol, runtime_checkable

import torch.nn as nn


class TokenizerProto(Protocol): ...


class ModelLoadingConfigBase[ModelT: nn.Module](SweepableConfig):
    model_name: str

    @cached_property
    def tokenizer(self) -> TokenizerProto:
        raise NotImplementedError

    def _make_raw_model(
        self,
        load_as_dtype: torch.dtype | None = None,
        device: str | torch.device = "cuda",
    ) -> ModelT:
        raise NotImplementedError

    def nnsight_wrap(self, model: ModelT) -> NNsight:
        raise NotImplementedError


# @runtime_checkable
# class ModelLoadingConfigBase(Protocol):
#     model_name: str

#     @cached_property
#     def tokenizer(self) -> TokenizerProto:
#         raise NotImplementedError

#     def _make_raw_model(
#         self,
#         load_as_dtype: torch.dtype | None = None,
#         device: str | torch.device = "cuda",
#     ):
#         raise NotImplementedError

#     def nnsight_wrap(self, model) -> NNsight:
#         raise NotImplementedError
