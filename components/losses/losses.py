import torch
import torch.nn as nn
from sae_components.components.sae_cache import SAECache
from torch import Tensor
from jaxtyping import Float
from sae_components.components.reused_forward import ReuseForward
from typing import Protocol, runtime_checkable, List
from abc import abstractmethod
import sae_components.core as cl

# Not sure about this, should return to after the structure of gated example is more pinned down


class Loss(nn.Module):
    def __init__(
        self,
        module,
        # loss_coefficient=1,
    ):
        super().__init__()
        self.module = ReuseForward(module)
        # self.loss_coefficient = loss_coefficient

    def forward(self, x, y=None, cache: SAECache = None):
        assert cache is not None
        pred = self.module(x, cache=cache)
        if y is None:
            y = x
        return self.loss(x, y, pred, cache)

    @abstractmethod
    def loss(self, x, y, y_pred, cache: SAECache): ...

    # def __mul__(self, other):
    #     self.loss_coefficient *= other
    #     return self.__class__(self.module, self.loss_coefficient * other)

    # def __imul__(self, other): ...  # and log the updated value?


class L2Loss(Loss):
    def loss(self, x, y, y_pred, cache: SAECache):
        return torch.mean((y - y_pred) ** 2)


class SparsityPenalty(Loss):
    def loss(self, x, y, y_pred, cache: SAECache):
        sparsity_losses = [c for c in cache.search("sparsity_penalty")]
        assert (
            len(sparsity_losses) == 1
        ), "Expected exactly one sparsity penalty. We may want to support >1 in the future, so this may or may not be a bug."


@runtime_checkable
class HasLosses(Protocol):
    losses: List[Loss]


# Maybe the losses should just be functions and take (model, x, cache) as arguments

# in trainer assert isinstance(model, HasLosses)
model = ...
mse = MSELoss(model)
l1 = L1Loss(model)
