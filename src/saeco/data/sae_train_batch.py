from functools import cached_property
from typing import ClassVar

import torch
from attrs import define
from typing_extensions import dataclass_transform

from saeco.data.dict_batch import DictBatch


@DictBatch.auto_other_fields
class SAETrainBatch(DictBatch):
    input_sites: list[str]
    target_sites: list[str] | None = None

    @cached_property  # this means we must never mutate one of these? so maybe property is better.
    # but re-catting input and output each time seems bad too.
    def input(self) -> torch.Tensor:
        return torch.cat([self[k] for k in self.input_sites], dim=-1)

    @cached_property
    def target(self) -> torch.Tensor:
        sites = self.target_sites or self.input_sites
        return torch.cat([self[k] for k in sites], dim=-1)
