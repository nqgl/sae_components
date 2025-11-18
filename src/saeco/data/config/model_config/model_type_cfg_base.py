from functools import cached_property

from nnsight import NNsight
from pydantic import Field
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
)

from saeco.data.config.locations import DATA_DIRS

from saeco.data.dict_batch import DictBatch
from saeco.sweeps import SweepableConfig
from typing import Protocol, runtime_checkable
from typing import Iterable
import torch.nn as nn


class TokenizerProto(Protocol): ...


class ModelLoadingConfigBase[
    ModelT: nn.Module = nn.Module,
](SweepableConfig):
    @property
    def name(self) -> str:  # type: ignore
        raise NotImplementedError

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

    def input_data_transform[InputDataT: torch.Tensor | DictBatch](
        self, input_data: InputDataT
    ) -> InputDataT:
        return input_data

    def custom_data_source(self) -> Iterable[DictBatch]:
        raise NotImplementedError

    def filter_acts(
        self, input_data: torch.Tensor | DictBatch, acts: DictBatch
    ) -> DictBatch:
        return acts


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
