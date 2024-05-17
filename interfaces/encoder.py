import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from sae_components.components.sae_cache import SAECache
from torch import Tensor
from jaxtyping import Float


class SAE(nn.Module, ABC):
    encoder: nn.Module
    decoder: nn.Module

    @abstractmethod
    def forward(self, x, cache: SAECache): ...


class Encoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, cache: SAECache): ...
