import torch
from abc import ABC, abstractmethod

import torch.nn as nn
from saeco.core import PassThroughModule


class FreqTracker(PassThroughModule, ABC):
    def __init__(self):
        super().__init__()
        self.is_active = True

    @property
    @abstractmethod
    def freqs(self) -> torch.Tensor: ...

    # freqs = self.activation_freqs
    # if freqs.ndim != 1:
    #     raise ValueError(
    #         f"Expected 1D freqs, got {freqs.ndim}D with shape {freqs.shape}.\
    #         If multidimensional freqs are correct for some use cases, will add support for it."
    #     )
    # return freqs
    @torch.no_grad()
    def process_data(self, acts, cache, **kwargs):
        if self.training:
            freqs = (acts != 0).float().mean(dim=0)
            self.update_freqs(freqs, cache=cache)

    @abstractmethod
    def update_freqs(self, freqs: torch.Tensor, cache): ...

    @abstractmethod
    def reset(self): ...


def get_freq_trackers(model: nn.Module):
    l = set()
    for m in model.modules():
        if isinstance(m, FreqTracker):
            l.add(m)
    return l


def get_active_freq_trackers(model: nn.Module):
    return {ft for ft in get_freq_trackers(model) if ft.is_active}
