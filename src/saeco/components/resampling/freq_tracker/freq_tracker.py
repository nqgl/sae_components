import torch
from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod
from saeco.core import PassThroughModule


class FreqTracker(PassThroughModule, ABC):
    @abstractmethod
    @property
    def activation_freqs(self) -> torch.Tensor: ...

    @property
    def freqs(self) -> torch.Tensor:
        freqs = self.activation_freqs
        if freqs.ndim != 1:
            raise ValueError(
                f"Expected 1D freqs, got {freqs.ndim}D with shape {freqs.shape}.
                If multidimensional freqs are correct for some use cases, will add support for it."
            )
        return freqs

    @abstractmethod
    def reset(self): ...
