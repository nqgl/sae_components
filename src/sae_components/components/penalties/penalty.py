from sae_components.components.sae_cache import SAECache


import torch.nn as nn
import sae_components.core as cl
from abc import abstractmethod


class Penalty(cl.PassThroughModule):

    def process_data(self, x, *, cache, **kwargs):
        cache.sparsity_penalty = self.penalty(x)

    @abstractmethod
    def penalty(self, x, *, cache, **kwargs): ...
