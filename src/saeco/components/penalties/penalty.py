from abc import abstractmethod

import saeco.core as cl
from saeco.components.sae_cache import SAECache


class Penalty(cl.PassThroughModule):
    def process_data(self, x, *, cache: SAECache, **kwargs):
        cache.sparsity_penalty = self.penalty(x, cache=cache)

    @abstractmethod
    def penalty(self, x, *, cache): ...


class LambdaPenalty(Penalty):
    def __init__(self, penalty):
        super().__init__()
        self._penalty = penalty

    def penalty(self, x, *, cache):
        return self._penalty(x)
