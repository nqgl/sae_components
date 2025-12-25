from abc import ABC, abstractmethod

import torch.nn as nn

from saeco.components.sae_cache import SAECache


class SAE(nn.Module, ABC):
    encoder: nn.Module
    decoder: nn.Module

    @abstractmethod
    def forward(self, x, *, cache: SAECache): ...


class Encoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, *, cache: SAECache): ...
