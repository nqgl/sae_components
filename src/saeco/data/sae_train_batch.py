from functools import cached_property
from typing import ClassVar

import torch
from attrs import define
from typing_extensions import dataclass_transform

from saeco.data.piler.dict_piler import DictBatch


# @dataclass_transform(kw_only_default=True)
@DictBatch.auto_other_fields
class SAETrainBatch(DictBatch):
    input_sites: list[str]
    target_sites: list[str] | None = None
    # OTHER_DATA_FIELDS: ClassVar[tuple[str, ...]] = DictBatch.OTHER_DATA_FIELDS + (
    #     "input_sites",
    #     "target_sites",
    # )

    @cached_property  # this means we must never mutate one of these? so maybe property is better.
    # but re-catting input and output each time seems bad too.
    def input(self) -> torch.Tensor:
        return torch.cat([self[k] for k in self.input_sites], dim=-1)

    @cached_property
    def target(self) -> torch.Tensor:
        sites = self.target_sites or self.input_sites
        return torch.cat([self[k] for k in sites], dim=-1)

    # def get_shapes(self, sites: list[str]):
    #     return [self[k].shape for k in sites]

    # def pack_tensor_like_inputs(self, tensor: torch.Tensor) -> torch.Tensor:

    #     shapes = [shape[-1] for shape in self.get_shapes(self.input_sites)]
    #     splits = [sum(shapes[: i + 1]) for i in range(len(shapes) - 1)]
    #     split = tensor.tensor_split(splits, dim=-1)
    #     d = {k: v for k, v in zip(self.input_sites, split, strict=True)}
    #     return self.
