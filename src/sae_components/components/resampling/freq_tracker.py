import torch
from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod
import sae_components.core as cl


# @runtime_checkable
# class FreqTracker(Protocol):
#     def update(self, acts: torch.Tensor): ...
#     @property
#     def freqs(self) -> torch.Tensor: ...
#     def reset(self): ...


class PassThrough(cl.Module):
    def forward(self, x, cache: cl.Cache = None, **kwargs):
        self.action(x, cache, **kwargs)
        return x

    def action(self, x, cache: cl.Cache = None, **kwargs):
        return x


class FreqTracker(ABC, cl.Module):
    def update(self, acts: torch.Tensor): ...
    @property
    def freqs(self) -> torch.Tensor: ...
    def reset(self): ...
