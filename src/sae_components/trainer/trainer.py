import sae_components.core.module as cl
from sae_components.core import Cache
from sae_components.components.sae_cache import SAECache
import torch
import wandb
from typing import Protocol, runtime_checkable
from sae_components.components.losses import Loss, L2Loss, SparsityPenaltyLoss
from dataclasses import dataclass
from sae_components.trainer.post_backward_normalization import post_backward, post_step
import torch.nn as nn

# @runtime_checkable
# class TrainableModel(Protocol, cl.Module):
#     losses: list[Loss]
#     model: cl.Module
#     models: dict[cl.Module]

#     def forward(self, x: torch.Tensor, cache: Cache = None) -> torch.Tensor: ...


class Trainable(cl.Module):
    losses: dict[Loss]
    model: cl.Module
    models: list[cl.Module]

    def __init__(self, models: list[cl.Module], losses: dict[Loss]):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.model = models[0]
        self.models = nn.ModuleList(models)

    def loss(self, x, *, cache, y=None, coeffs={}):
        coeffs = dict(coeffs)
        loss = 0
        for k, L in self.losses.items():
            loss += L(x, y=y, cache=cache[k]) * coeffs.pop(k, 1)
        cache.loss = loss.item()
        assert len(coeffs) == 0, f"loss coefficient cfg had unused keys: {coeffs}"
        return loss

    def forward(self, x: torch.Tensor, cache: Cache = None) -> torch.Tensor: ...


@dataclass
class TrainConfig: ...


class TrainCache(SAECache):
    L2_loss = ...
    sparsity_penalty = ...
    L2_aux_loss = ...
    loss = ...
    L1 = ...
    L0 = ...
    L1_full = ...
    L0_aux = ...


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: Trainable,
        # optim: torch.optim.Optimizer,
    ):
        self.cfg = cfg
        self.model = model
        # self.sae.provide("optim", self.optim)
        self.t = 1
        self.extra_calls = []
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def post_backward(self):
        self.model.apply(post_backward)

    def post_step(self):
        self.model.apply(post_step)

    def coeffs(self):
        # TODO
        return {"sparsisty_loss": 0.02}

    def train(self, buffer):
        self.post_step()
        for bn in buffer:
            if isinstance(bn, tuple):
                x, y = bn
            else:
                x = bn
                y = x

            cache = TrainCache()
            loss = self.model.loss(x, cache=cache, coeffs=self.coeffs())
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
