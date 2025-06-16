from functools import cached_property
from typing import ClassVar

import torch
from attrs import define

from saeco.data.piler.dict_piler import DictBatch


@define
class SAETrainBatch(DictBatch):
    input_sites: list[str]
    target_sites: list[str]
    OTHER_DATA_FIELDS: ClassVar[tuple[str, ...]] = DictBatch.OTHER_DATA_FIELDS + (
        "input_sites",
        "target_sites",
    )

    @cached_property  # this means we must never mutate one of these? so maybe property is better.
    # but re-catting input and output each time seems bad too.
    def input(self) -> torch.Tensor:
        return torch.cat([self[k] for k in self.input_sites], dim=-1)

    @cached_property
    def target(self) -> torch.Tensor:
        return torch.cat([self[k] for k in self.target_sites], dim=-1)
