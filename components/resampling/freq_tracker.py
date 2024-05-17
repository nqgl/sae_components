import torch
from typing import Protocol, runtime_checkable


@runtime_checkable
class FreqTracker(Protocol):
    def update(self, acts: torch.Tensor): ...
    @property
    def freqs(self) -> torch.Tensor: ...
    def reset(self): ...
