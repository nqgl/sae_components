import torch
from typing import Protocol, runtime_checkable
from abc import ABC
from saeco.core import PassThroughModule


class FreqTracker(PassThroughModule):
    @property
    def freqs(self) -> torch.Tensor: ...
    def reset(self): ...
