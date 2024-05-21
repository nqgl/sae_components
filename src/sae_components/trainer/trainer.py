import sae_components.core as cl
from sae_components.core import Cache
import torch
import wandb
from typing import Protocol, runtime_checkable, List

from sae_components.components.losses import Loss, L2Loss, SparsityPenaltyLoss
from dataclasses import dataclass


@runtime_checkable
class TrainableModel(Protocol, cl.Module):
    losses: List[Loss]
    model: cl.Module

    def forward(self, x: torch.Tensor, cache: Cache = None) -> torch.Tensor: ...


@dataclass
class TrainConfig: ...


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: TrainableModel,
        optim: torch.optim.Optimizer,
    ):
        self.cfg = cfg
        self.model = model
        # self.sae.provide("optim", self.optim)
        self.t = 1
        self.extra_calls = []
        self.optim = optim

    # def update_optim_lrs(self):
    #     for pg in self.optim.param_groups:
    #         if pg["name"] == "bias":
    #             pg["lr"] = self.cfg.lr * self.cfg.bias_lr_coeff
    #             if getattr(self.cfg, "weight_decay_bias", None):
    #                 pg["weight_decay"] = (
    #                     self.cfg.weight_decay_bias or self.cfg.weight_decay
    #                 )
    #         elif pg["name"] == "weight":
    #             pg["lr"] = self.cfg.lr
    #             if getattr(self.cfg, "weight_decay", None):
    #                 pg["weight_decay"] = self.cfg.weight_decay

    #         else:
    #             raise ValueError(f"param group name {pg['name']} not recognized")

    # def parameters(self):
    #     bias_params = []
    #     for name, param in self.sae.named_parameters():
    #         if name.endswith(".b_dec") or name.endswith(".b") or name.endswith(".bias"):
    #             bias_params.append(param)

    #     weights = set(self.sae.parameters()) - set(bias_params)
    #     groups = [
    #         {
    #             "params": bias_params,
    #             "lr": self.cfg.lr * self.cfg.bias_lr_coeff,
    #             "name": "bias",
    #         },
    #         {"params": list(weights), "lr": self.cfg.lr, "name": "weight"},
    #     ]
    #     return groups

    def loss(self, x):
        loss = 0
        for L in self.model.losses:
            loss += L(x)

    def train(self, buffer):
        self.norm_dec()
        for bn in buffer:
            try:
                x, y = bn
                assert isinstance(bn, tuple)
            except:
                y = (x := bn)
                assert isinstance(bn, torch.Tensor)
            # in the future trainer could itself be a CacheProcLayer maybe
            cache: SAETrainCache = self.sae.generate_default_cache()
            y_pred = self.sae(x, cache=cache)
            cache.y = ...
            cache.y = y
            self.step(cache=cache, x=x, y_pred=y_pred, y=x)
            self.full_log(cache)
            self.t += 1

    def step(self, cache: SAETrainCache, x, y_pred, y=None):
        loss = self.loss(cache)
        cache.loss = loss
        loss.backward()
        resample_before_step = getattr(
            self.cfg.resampler_cfg, "resample_before_step", False
        )
        if resample_before_step:
            for c in cache.search("resample"):
                c.num_resampled = ...
                c.resample(x=x)
        self.orthogonalize_dec_grads()
        self.optim.step()
        self.optim.zero_grad()
        self.norm_dec()
        if not resample_before_step:
            for c in cache.search("resample"):
                c.num_resampled = ...
                c.resample(x=x)
        for call in self.extra_calls:
            call(cache)

    def full_log(self, cache: Cache):
        if self.t % 10 != 0:
            return
        d = cache.logdict(
            excluded=[
                "acts",
                "y_pred",
                "x",
                "y",
                "resample",
                "nonlinear_argsmaxed",
                "acts_spoof",
            ]
        )
        if wandb.run is not None:
            wandb.log(d, step=self.t)

    def save(self):
        torch.save(
            self.sae.state_dict(), "/root/workspace/" + wandb.run.name + f"_{self.t}.pt"
        )


class SAETrainer(Trainer):
    @torch.no_grad()
    def orthogonalize_dec_grads(
        self,
    ):  # TODO move to SAEComponentLayer and use callback
        if getattr(self.cfg, "norm_dec_grads", True):
            grad = self.sae.cachelayer.decoder.weight.grad
            dec_normed = (
                self.sae.cachelayer.decoder.weight.data
                / self.sae.cachelayer.decoder.weight.data.norm(dim=0, keepdim=True)
            )
            grad_orth = grad - (dec_normed * grad).sum(0, keepdim=True) * dec_normed
            self.sae.cachelayer.decoder.weight.grad[:] = grad_orth

    @torch.no_grad()
    def norm_dec(self):  # TODO move to SAEComponentLayer and use callback
        if self.t % 1000 == 0:
            wandb.log(
                {
                    "dec_norms": self.sae.cachelayer.decoder.weight.norm(dim=0)
                    .max()
                    .item()
                },
                step=self.t,
            )
        if self.cfg.actual_tied_weights:
            return
        norm = self.sae.cachelayer.decoder.weight.data.norm(dim=0, keepdim=True)
        normed = self.sae.cachelayer.decoder.weight.data / norm
        self.sae.cachelayer.decoder.weight.data[:] = (
            torch.where(norm > 1, normed, self.sae.cachelayer.decoder.weight.data)
            if self.cfg.selectively_norm_dec
            else normed
        )

    def loss(
        self, cache: SAETrainCache
    ):  # TODO replace with slot for a fn that does Cache, Config->Number
        return (
            self.get_l2_type(cache, self.cfg.l2_loss_type)
            + cache["encoder"].l1 * self.cfg.l1_coeff
            + (
                cache["encoder"].l0l1 * self.cfg.l0l1_coeff
                if self.cfg.l0l1_coeff and self.cfg.l0l1_coeff != 0
                else 0
            )
        )

    def get_l2_type(self, cache, l2_type):
        if isinstance(l2_type, str):
            if l2_type == "squared/40":
                return cache.l2 / 40
            elif l2_type == "l2_norm":
                return cache.l2_norm
            elif l2_type == "l2_root":
                return cache.l2**0.5
            elif l2_type == "l2_norm_squared/40":
                return cache.l2_norm**2 / 40
            else:
                raise ValueError(f"l2_type {l2_type} not recognized")
        else:
            v = 0
            for l2 in l2_type:
                v = v + self.get_l2_type(cache, l2)
            return v / len(l2_type)
