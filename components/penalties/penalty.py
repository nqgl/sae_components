from sae_components.components.sae_cache import SAECache


import torch.nn as nn


from abc import abstractmethod


class Penalty(nn.Module):
    def forward(self, x, cache: SAECache):
        cache.sparsity_penalty = self.penalty(x)
        return x

    @abstractmethod
    def penalty(self, x): ...
