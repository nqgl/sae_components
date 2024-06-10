from saeco.components.sae_cache import SAECache


import torch.nn as nn
import saeco.core as cl
from abc import abstractmethod


class Penalty(cl.PassThroughModule):

    def process_data(self, x, *, cache: SAECache, **kwargs):
        cache.sparsity_penalty = self.penalty(x)

    @abstractmethod
    def penalty(self, x): ...


class LambdaPenalty(Penalty):
    def __init__(self, penalty):
        super().__init__()
        self._penalty = penalty

    def penalty(self, x):
        return self._penalty(x)
