from abc import abstractmethod
from collections.abc import Callable

from torch import Tensor

import saeco.core as cl
from saeco.components.sae_cache import SAECache


class Penalty(cl.PassThroughModule):
    def process_data(self, x, *, cache: SAECache, **kwargs) -> None:
        cache.sparsity_penalty = self.penalty(x, cache=cache)

    @abstractmethod
    def penalty(self, x, *, cache) -> Tensor: ...


class LambdaPenalty(Penalty):
    def __init__(self, penalty: Callable[[Tensor], Tensor]):
        super().__init__()
        self._penalty = penalty

    def penalty(self, x, *, cache) -> Tensor:
        return self._penalty(x)
