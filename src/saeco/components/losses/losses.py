from abc import abstractmethod
from typing import List, Protocol, runtime_checkable

import torch

import saeco.core as cl
from saeco.components.sae_cache import SAECache
from saeco.core.reused_forward import ReuseForward


class Loss(cl.Module):
    def __init__(
        self,
        module,
    ):
        super().__init__()
        self.module = ReuseForward(module)

    def forward(self, x, *, y=None, cache: cl.Cache, **kwargs) -> torch.Tensor:
        assert cache is not None
        pred = self.module(x, cache=cache)
        if y is None:
            y = x
        return self.loss(x, y, pred, cache)

    @abstractmethod
    def loss(self, x, y, y_pred, cache: SAECache) -> torch.Tensor: ...


class L2Loss(Loss):
    def loss(self, x, y, y_pred, cache: SAECache):
        return torch.mean((y - y_pred) ** 2)


class SparsityPenaltyLoss(Loss):
    def __init__(self, module, num_expeted=1):
        self.num_expected = num_expeted
        super().__init__(module)

    def loss(self, x, y, y_pred, cache: SAECache):
        sparsity_losses = cache._ancestor.search("sparsity_penalty")
        assert (
            len(sparsity_losses) == self.num_expected
        ), f"Expected exactly one (or self.num_expected) sparsity penalty, but got {len(sparsity_losses)}. We may want to support >1 in the future, so this may or may not be a bug."
        l = 0
        for sp_cache in sparsity_losses:
            l += sp_cache.sparsity_penalty.squeeze()
        return l


class CosineSimilarityLoss(Loss):
    def loss(self, x, y, y_pred, cache: SAECache):
        return torch.cosine_similarity(y, y_pred, dim=-1).mean()


class TruncatedLoss:
    def __init__(self, *args, truncate_at: int, **kwargs):
        self.truncate_at = truncate_at
        super().__init__(*args, **kwargs)
        self.module = self.module.module

    def forward(self, x, *, y=None, cache: cl.Cache, **kwargs):
        assert cache is not None
        old_cache = cache
        cache = cache.clone()
        cache._ancestor.forward_reuse_dict = old_cache._ancestor.forward_reuse_dict

        def acts_callback(cache: cl.Cache, acts):
            if not cache.has.act_metrics_name:
                return acts
            if cache._parent is None:
                return acts
            if (
                cache.act_metrics_name is not None
            ):  # None corresponds to main activations
                return acts
            z = torch.zeros_like(acts)
            z[:, : self.truncate_at] = acts[:, : self.truncate_at]
            return z

        cache.register_write_callback("acts", acts_callback)
        pred = self.module(x, cache=cache)
        if y is None:
            y = x
        l = self.loss(x, y, pred, cache)
        cache.destruct()
        return l


class TruncatedL2Loss(TruncatedLoss, L2Loss):
    pass
