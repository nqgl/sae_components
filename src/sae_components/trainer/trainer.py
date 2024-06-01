import sae_components.core as cl
from sae_components.core import Cache
from sae_components.components.sae_cache import SAECache
import torch
import wandb
from typing import Protocol, runtime_checkable, Optional
from sae_components.components.losses import Loss, L2Loss, SparsityPenaltyLoss
from dataclasses import dataclass, field
from sae_components.trainer.post_backward_normalization import post_backward, post_step
from sae_components.trainer.normalizers import Normalizer, ConstL2Normalizer, Normalized
import torch.nn as nn


@dataclass
class OptimConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.99)


@dataclass
class TrainConfig:
    coeffs: dict[str, float] = field(default_factory=lambda: dict(sparsity_loss=3e-3))
    optim_config: OptimConfig = OptimConfig()
    l0_target: Optional[float] = None
    l0_target_adjustment_size: float = 0.0001


class TrainCache(SAECache):
    L2_loss = ...
    sparsity_penalty = ...
    L2_aux_loss = ...
    loss = ...
    L1 = ...
    L0 = ...
    L1_full = ...
    L0_aux = ...


class Trainable(cl.Module):

    losses: dict[Loss]
    model: cl.Module
    models: list[cl.Module]
    normalizer: Normalizer

    def __init__(
        self,
        models: list[cl.Module],
        losses: dict[Loss],
        normalizer: Normalizer = None,
    ):
        super().__init__()
        self.normalizer = normalizer or ConstL2Normalizer()
        assert not any(
            isinstance(m, Normalized) for m in list(losses.values()) + models
        ), "models and losses should not be normalized, the Trainable object is responsible for normalization."
        self.losses = nn.ModuleDict(
            {
                name: self.normalizer.input_normalize(loss)
                for name, loss in losses.items()
            }
        )

        self.model = self.normalizer.io_normalize(models[0])
        self.models = nn.ModuleList(models)

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
        return self.model(x, cache=cache)


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: Trainable,
        namestuff=None,
        # optim: torch.optim.Optimizer,
    ):
        self.cfg = cfg
        self.model = model
        # self.sae.provide("optim", self.optim)
        wandb.init(
            project="sae-components",
            config={"model": repr(model), "cfg": cfg},
            reinit=True,
        )
        if namestuff is not None:
            wandb.run.name = (
                f"{namestuff}[{cfg.l0_target}]-{wandb.run.name.split('-')[-1]}"
            )
        self.t = 1
        self.extra_calls = []
        self.optim = torch.optim.RAdam(
            self.model.parameters(), lr=3e-4, betas=(0.9, 0.99)
        )

    def post_backward(self):
        self.model.apply(post_backward)

    def post_step(self):
        self.model.apply(post_step)

    def log(self, d):
        if wandb.run is not None:
            wandb.log(d, step=self.t)

    def coeffs(self):
        # TODO
        return self.cfg.coeffs

    def proc_cache_after_forward(self, cache: TrainCache):
        if self.cfg.l0_target is not None:
            self.cfg.coeffs["sparsity_loss"] = self.cfg.coeffs["sparsity_loss"] * (
                1
                + (-1 if self.cfg.l0_target > cache.L0 else 1)
                * self.cfg.l0_target_adjustment_size
            )
            self.log({"dynamic_sparsity_coeff": self.cfg.coeffs["sparsity_loss"]})

    def train(self, buffer):
        if self.t <= 1:
            self.model.normalizer.prime_normalizer(buffer)
        self.post_step()
        for bn in buffer:
            if isinstance(bn, tuple):
                x, y = bn
            else:
                x = bn
                y = x

            cache = TrainCache()
            loss = self.model.loss(x, cache=cache, coeffs=self.coeffs())
            self.proc_cache_after_forward(cache)
            loss.backward()
            self.post_backward()
            self.optim.step()
            self.post_step()
            self.optim.zero_grad()
            self.full_log(cache)
            self.t += 1
            print(f"loss: {loss.item()}")
            del cache.forward_reuse_dict
            cache.destroy_children()
            del cache
            # for key in [k for k in cache.forward_reuse_dict.keys()]:
            #     del cache.forward_reuse_dict[key]
            #     del key
            # del x, y, loss, cache

    def full_log(self, cache: Cache):
        if self.t % 10 != 0:
            return
        # d = cache.logdict(
        #     excluded=[
        #         "acts",
        #         "y_pred",
        #         "x",
        #         "y",
        #         "resample",
        #         "nonlinear_argsmaxed",
        #         "acts_spoof",
        #     ]
        # )
        if wandb.run is not None:
            wandb.log(cache.logdict(), step=self.t)

    def save(self):
        torch.save(
            self.sae.state_dict(), "/root/workspace/" + wandb.run.name + f"_{self.t}.pt"
        )
