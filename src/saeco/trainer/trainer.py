import torch.utils
from saeco.core import Cache
import torch
import wandb
from typing import Protocol, runtime_checkable, Optional
from saeco.components.losses import L2Loss, SparsityPenaltyLoss
from dataclasses import dataclass, Field
from saeco.data.sc.model_cfg import ModelConfig
from saeco.trainer.train_cache import TrainCache
from saeco.trainer.trainable import Trainable
from saeco.trainer.post_backward_normalization import (
    do_post_backward,
    do_post_step,
)
from .recons import get_recons_loss
from saeco.data.sc.dataset import DataConfig, SplitConfig, TokensData
from saeco.sweeps import SweepableConfig
from pydantic import Field


@dataclass
class OptimConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)


class TrainConfig(SweepableConfig):
    data_cfg: DataConfig = Field(default_factory=DataConfig)
    wandb_cfg: dict = Field(default_factory=lambda: dict(project="sae-components"))
    coeffs: dict[str, float] = Field(default_factory=lambda: dict(sparsity_loss=1e-3))
    l0_target: Optional[float] = None
    l0_target_adjustment_size: float = 0.0003
    use_autocast: bool = True
    batch_size: int = 4096
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    use_lars: bool = False
    kwargs: dict = Field(default_factory=dict)
    run_length: Optional[int] = 100e3
    resample_freq: int = 1500


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: Trainable,
        namestuff=None,
        optim: torch.optim.Optimizer = None,
    ):
        self.cfg: TrainConfig = cfg
        self.model = model
        if wandb.run is None:
            wandb.init(
                **cfg.wandb_cfg,
                config={"model": repr(model), "cfg": cfg},
                reinit=True,
            )
        if namestuff is not None:
            lars = "(lars)" if cfg.use_lars else ""
            wandb.run.name = (
                f"{lars}{namestuff}[{cfg.l0_target}]-{wandb.run.name.split('-')[-1]}"
            )
        self.t = 1
        self.extra_calls = []
        self.optim = optim or torch.optim.RAdam(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
        )
        if self.cfg.use_lars:
            from torchlars import LARS

            self.optim = LARS(self.optim)
            assert optim is None or not isinstance(optim, LARS)

        self.llm_val_tokens = TokensData(
            self.cfg.data_cfg, self.subject_model
        ).get_tokens_from_split(self.cfg.data_cfg.testsplit)
        self.intermittent_metric_freq = 1000
        self.gradscaler = torch.cuda.amp.GradScaler() if self.cfg.use_autocast else None

    @property
    def subject_model(self):
        return self.cfg.data_cfg.model_cfg.model

    def post_backward(self):
        do_post_backward(self.model)

    def post_step(self):
        do_post_step(self.model)

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

    def get_databuffer(self, num_batches=None, num_workers=0):
        return self.cfg.data_cfg.get_databuffer(
            num_workers=num_workers, batch_size=self.cfg.batch_size
        )

    def get_cache(self):
        c = TrainCache()
        c._watch(self.model.get_losses_and_metrics_names())
        # for k in self.model.get_losses_and_metrics_names():
        # setattr(c, k, ...)
        return c

    def train(self, buffer=None):
        if buffer is None:
            buffer = self.get_databuffer(num_workers=0)
        if self.t <= 1:
            self.model.normalizer.prime_normalizer(buffer)
        self.post_step()

        for x in buffer:
            x = x.cuda()
            if not self.cfg.use_autocast:
                x = x.float()
            self.optim.zero_grad()

            cache = self.get_cache()
            if self.cfg.use_autocast:
                with torch.autocast(device_type="cuda"):
                    loss = self.model.loss(x, cache=cache, coeffs=self.coeffs())
                    self.proc_cache_after_forward(cache)

            else:
                loss = self.model.loss(x, cache=cache, coeffs=self.coeffs())
                self.proc_cache_after_forward(cache)

            if self.cfg.use_autocast:
                self.gradscaler.scale(loss).backward()
            else:
                loss.backward()
            self.post_backward()

            if self.cfg.use_autocast:
                self.gradscaler.step(self.optim)
                self.gradscaler.update()
            else:
                self.optim.step()
            self.post_step()
            self.full_log(cache)
            self.t += 1
            del cache.forward_reuse_dict
            cache.destroy_children()
            del cache
            if self.t % self.intermittent_metric_freq == 0:
                self.do_intermittent_metrics()
            if self.t % self.cfg.resample_freq == 0:
                self.model.resampler.resample(buffer)
            if self.cfg.run_length and self.t > self.cfg.run_length:
                break

    def do_intermittent_metrics(self):
        self.log_recons("recons/with_bos/", True)
        self.log_recons("recons/no_bos/", False)

    def log_recons(self, label, proc_bos, num_batches=5, dumb_rescaled=False):
        def run_rescaled_model(x):
            cache = self.get_cache()
            cache.scale = ...
            cache.scale = True
            return self.model(x, cache=cache)

        self.log(
            {
                (label + k): v
                for k, v in get_recons_loss(
                    self.subject_model,
                    run_rescaled_model if dumb_rescaled else self.model,
                    buffer=None,
                    all_tokens=self.llm_val_tokens,
                    cfg=self.cfg.data_cfg.model_cfg.acts_cfg,
                    bos_processed_with_hook=proc_bos,
                    num_batches=num_batches,
                ).items()
            }
        )

    def full_log(self, cache: Cache):
        if self.t % 10 != 0:
            return
        if wandb.run is not None:
            wandb.log(cache.logdict(), step=self.t)

    def save(self):
        torch.save(
            self.sae.state_dict(), "/root/workspace/" + wandb.run.name + f"_{self.t}.pt"
        )
