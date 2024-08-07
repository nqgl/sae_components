import torch
from saeco.components.sae_cache import SAECache
from saeco.core.reused_forward import ReuseForward
from typing import Protocol, runtime_checkable, List
from abc import abstractmethod
import saeco.core as cl

# Not sure about this, should return to after the structure of gated example is more pinned down


class Loss(cl.Module):
    def __init__(
        self,
        module,
        # loss_coefficient=1,
    ):
        super().__init__()
        self.module = ReuseForward(module)
        # self.loss_coefficient = loss_coefficient

    def forward(self, x, *, y=None, cache: cl.Cache, **kwargs):
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


class SparsityPenaltyLoss(Loss):
    def __init__(self, module, num_expeted=1):
        self.num_expected = num_expeted
        super().__init__(module)

    def loss(self, x, y, y_pred, cache: SAECache):
        sparsity_losses = [c for c in cache._ancestor.search("sparsity_penalty")]
        assert (
            len(sparsity_losses) == self.num_expected
        ), f"Expected exactly one (or self.num_expected) sparsity penalt(y/ies), but got {len(sparsity_losses)}. We may want to support >1 in the future, so this may or may not be a bug."
        l = 0
        for sp_cache in sparsity_losses:
            l += sp_cache.sparsity_penalty.squeeze()
        return l


@runtime_checkable
class HasLosses(Protocol):
    losses: List[Loss]


class CosineSimilarityLoss(Loss):
    def loss(self, x, y, y_pred, cache: SAECache):
        return torch.cosine_similarity(y, y_pred, dim=-1).mean()


# # Maybe the losses should just be functions and take (model, x, cache) as arguments

# # in trainer assert isinstance(model, HasLosses)
# model = ...
# mse = MSELoss(model)
# l1 = L1Loss(model)
