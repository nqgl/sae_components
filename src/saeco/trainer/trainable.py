import saeco.core as cl
from saeco.components.losses import (
    Loss,
    L2Loss,
    SparsityPenaltyLoss,
    CosineSimilarityLoss,
)
from saeco.core import Cache
from saeco.trainer.normalizers import ConstL2Normalizer, Normalized, Normalizer
from saeco.trainer.train_cache import TrainCache


import torch
import torch.nn as nn


class Trainable(cl.Module):

    losses: dict[Loss]
    model: cl.Module
    models: list[cl.Module]
    normalizer: Normalizer

    def __init__(
        self,
        models: list[cl.Module],
        losses: dict[str, Loss] = None,
        extra_losses: dict[str, Loss] = None,
        metrics: dict[str, Loss] = None,
        normalizer: Normalizer = None,
    ):
        super().__init__()
        self.normalizer = normalizer or ConstL2Normalizer()
        assert not any(
            isinstance(m, Normalized) for m in list(losses.values()) + models
        ), "models and losses should not be normalized, the Trainable object is responsible for normalization."
        self.models = nn.ModuleList(models)
        model = models[0]
        losses = losses or {
            "L2_loss": L2Loss(model),
            "sparsity_loss": SparsityPenaltyLoss(model),
        }
        assert extra_losses is None or losses is None
        self.losses = nn.ModuleDict(
            {
                **{
                    name: self.normalizer.input_normalize(loss)
                    for name, loss in losses.items()
                },
                **(extra_losses or {}),
            }
        )
        metrics = metrics or {"cosim": CosineSimilarityLoss(model)}
        if not "L2_loss" in self.losses:
            metrics["L2_loss"] = L2Loss(model)
        self.metrics = nn.ModuleDict(
            {
                name: self.normalizer.input_normalize(metric)
                for name, metric in metrics.items()
            }
        )
        self.model = self.normalizer.io_normalize(models[0])

    def _normalizeIO(mth):
        def wrapper(self, x: torch.Tensor, *, cache: TrainCache, **kwargs):
            x = self.normalizer(x)
            return mth(self, x, cache=cache, **kwargs)

        return wrapper

    def loss(self, x, *, cache: TrainCache, y=None, coeffs={}):
        coeffs = dict(coeffs)
        loss = 0
        for k, L in self.losses.items():
            m = L(x, y=y, cache=cache[k])
            setattr(cache, k, m)
            loss += m * coeffs.pop(k, 1)
        cache.loss = loss.item()
        # with torch.no_grad():
        for k, L in self.metrics.items():
            m = L(x, y=y, cache=cache[k]) * coeffs.pop(k, 1)
            setattr(cache, k, m)

        assert len(coeffs) == 0, f"loss coefficient cfg had unused keys: {coeffs}"
        return loss

    def forward(self, x: torch.Tensor, cache: Cache = None) -> torch.Tensor:
        made_cache = False
        if cache is None:
            cache = TrainCache()
            made_cache = True
        out = self.model(x, cache=cache)
        if made_cache:
            cache.destroy_children()
            del cache
        return out

    def get_losses_and_metrics_names(self) -> list[str]:
        return list(self.losses.keys()) + list(self.metrics.keys())
