import sae_components.core as cl
from sae_components.components.losses import Loss, L2Loss, SparsityPenaltyLoss
from sae_components.core import Cache
from sae_components.trainer.normalizers import ConstL2Normalizer, Normalized, Normalizer
from sae_components.trainer.train_cache import TrainCache


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
        losses: dict[Loss] = None,
        extra_losses: dict[Loss] = None,
        normalizer: Normalizer = None,
    ):
        super().__init__()
        self.normalizer = normalizer or ConstL2Normalizer()
        assert not any(
            isinstance(m, Normalized) for m in list(losses.values()) + models
        ), "models and losses should not be normalized, the Trainable object is responsible for normalization."
        self.model = self.normalizer.io_normalize(models[0])
        self.models = nn.ModuleList(models)
        losses = losses or {
            "l2_loss": L2Loss(self.model),
            "sparsity_loss": L2Loss(self.model),
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

    def _normalizeIO(mth):
        def wrapper(self, x: torch.Tensor, *, cache: TrainCache, **kwargs):
            x = self.normalizer(x)
            return mth(self, x, cache=cache, **kwargs)

        return wrapper

    def loss(self, x, *, cache: TrainCache, y=None, coeffs={}):
        coeffs = dict(coeffs)
        loss = 0
        for k, L in self.losses.items():
            l = L(x, y=y, cache=cache[k]) * coeffs.pop(k, 1)
            setattr(cache, k, l)
            loss += l
        cache.loss = loss.item()
        assert len(coeffs) == 0, f"loss coefficient cfg had unused keys: {coeffs}"
        return loss

    def forward(self, x: torch.Tensor, cache: Cache = None) -> torch.Tensor:
        if cache is None:
            cache = TrainCache()
        return self.model(x, cache=cache)
