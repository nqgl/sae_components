from sae_components.components.sae_cache import SAECache


import torch.nn as nn
import sae_components.core as cl
from abc import abstractmethod


class Penalty(cl.Module):
    def forward(self, x, cache: SAECache):
        cache.sparsity_penalty = self.penalty(x)
        return x

    @abstractmethod
    def penalty(self, x): ...
