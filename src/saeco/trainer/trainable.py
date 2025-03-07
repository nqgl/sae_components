from typing import Optional

import einops

import torch
import torch.nn as nn

import saeco.core as cl
from saeco.components.losses import (
    CosineSimilarityLoss,
    L2Loss,
    Loss,
    SparsityPenaltyLoss,
)
from saeco.components.resampling import AnthResampler, RandomResampler, Resampler
from saeco.core import Cache
from .normalizers import ConstL2Normalizer, Normalized, Normalizer
from .train_cache import TrainCache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saeco.architecture.architecture import SAE

from functools import wraps


def make_cache_optional(fn):
    @wraps(fn)
    def wrapper(self, *args, cache: TrainCache | None = None, **kwargs):
        made_cache = False
        if cache is None:
            cache = TrainCache()
            made_cache = True

        out = fn(self, *args, cache=cache, **kwargs)

        if made_cache:
            cache.destruct()
            del cache
        return out

    return wrapper


class Trainable(cl.Module):
    """
    current responsibilities:
    normalization:
        this handles normalization for both model io
        model losses in particular
        holds and can create the normalizer
         - not responsible for priming the normalizer
    sets up the resampling and holds the resampler
    losses/metrics:
        generate the total loss from all losses
    creates cache if no cache is provided (operability with external code)


    """

    losses: dict[Loss]
    model: cl.Module
    models: list[cl.Module]
    normalizer: Normalizer

    def __init__(
        self,
        models: list["SAE"],
        losses: Optional[dict[str, Loss]] = None,
        extra_losses: Optional[dict[str, Loss]] = None,
        metrics: Optional[dict[str, Loss]] = None,
        normalizer: Optional[Normalizer] = None,
        resampler: Optional[Resampler | bool] = None,
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
        self.encode_only = self.normalizer.input_normalize(models[0].encoder)
        self.decode_only = self.normalizer.output_denormalize(models[0].decoder)
        if resampler:
            self.resampler = resampler
        elif resampler is None:
            self.resampler = AnthResampler()
            self.resampler.assign_model(self.model)
        else:
            self.resampler = None

    def _normalizeIO(mth):
        def wrapper(self, x: torch.Tensor, *, cache: TrainCache, **kwargs):
            x = self.normalizer(x)
            return mth(self, x, cache=cache, **kwargs)

        return wrapper

    def loss(self, x, *, cache: TrainCache, y=None, coeffs={}):
        coeffs = dict(coeffs)
        loss = 0
        for key, loss_fn in self.losses.items():
            m = loss_fn(x, y=y, cache=cache[key])
            setattr(cache, key, m)
            loss += m * coeffs.pop(key, 1)
        cache.loss = loss.item()
        for key, metric_fn in self.metrics.items():
            m = metric_fn(x, y=y, cache=cache[key]) * coeffs.pop(key, 1)
            setattr(cache, key, m)

        assert len(coeffs) == 0, f"loss coefficient cfg had unused keys: {coeffs}"
        return loss

    def rearrange(self, x: torch.Tensor):
        shape = None
        if x.ndim == 3:
            shape = x.shape
            x = einops.rearrange(x, "doc seq data -> (doc seq) data")
        elif x.ndim != 2:
            raise NotImplementedError
        return x, shape

    def dearrange(self, x: torch.Tensor, shape: tuple[int, int]):
        if shape is None:
            return x
        return einops.rearrange(
            x, "(doc seq) data -> doc seq data", doc=shape[0], seq=shape[1]
        )

    @make_cache_optional
    def forward(self, x: torch.Tensor, *, cache: Cache) -> torch.Tensor:
        x, shape = self.rearrange(x)
        out = self.model(x, cache=cache)
        return self.dearrange(out, shape)

    def get_losses_and_metrics_names(self) -> list[str]:
        return (
            list(self.losses.keys())
            + list(self.metrics.keys())
            + ["below_3e-5", "below_1e-5", "below_3e-6", "below_1e-6"]
        )

    def param_groups(self, optim_kwargs: dict) -> list[dict]:
        from saeco.components.features.param_metadata import (
            MetaDataParam,
            ParamMetadata,
        )

        normal = []
        has_metadata = {}
        for name, param in self.named_parameters():
            if isinstance(param, MetaDataParam):
                md = param._param_metadata
                if not md.has_param_group_values():
                    normal.append(param)
                    continue
                key = tuple(md.param_group_values(optim_kwargs).items())
                if key not in has_metadata:
                    has_metadata[key] = []
                has_metadata[key].append(param)
            else:
                normal.append(param)
        groups = [{"name": "normal", "params": normal}]
        for kvs, params in has_metadata.items():
            groups.append({"params": params, **{k: v for k, v in kvs}})
        assert sum(len(g["params"]) for g in groups) == len(
            list(self.parameters())
        ), f"param_groups did not cover all parameters"
        return groups

    def make_cache(self) -> TrainCache:
        return TrainCache()

    @make_cache_optional
    def get_acts(self, x, cache: TrainCache = None, pre_acts=False):
        cache.acts = ...
        if pre_acts:
            cache.pre_acts = ...
        cache(self)(x)
        acts = cache.acts
        if pre_acts:
            preacts = cache.pre_acts
        if pre_acts:
            return acts, preacts
        return acts

    @make_cache_optional
    def encode(self, x, cache: TrainCache = None):
        x, shape = self.rearrange(x)
        return self.dearrange(cache(self).encode_only(x), shape)

    @make_cache_optional
    def decode(self, x, cache: TrainCache = None):
        x, shape = self.rearrange(x)
        return self.dearrange(cache(self).decode_only(x), shape)
