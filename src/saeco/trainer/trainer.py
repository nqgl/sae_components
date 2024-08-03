import torch.utils
from saeco.core import Cache
import torch
import wandb
from typing import Protocol, runtime_checkable, Optional
from saeco.components.losses import L2Loss, SparsityPenaltyLoss
from dataclasses import Field
from saeco.data.sc.model_cfg import ModelConfig
from saeco.trainer.OptimConfig import OptimConfig, get_optim_cls
from saeco.trainer.schedule_cfg import RunSchedulingConfig
from saeco.trainer.train_cache import TrainCache
from saeco.trainer.trainable import Trainable
from saeco.trainer.post_backward_normalization import (
    do_post_backward,
    do_post_step,
)
from .recons import get_recons_loss
from saeco.data.sc.dataset import DataConfig, SplitConfig, TokensData
from saeco.sweeps import SweepableConfig, Swept
from pydantic import Field
from .l0targeter import L0Targeter
from saeco.misc import lazycall
from schedulefree import ScheduleFreeWrapper, AdamWScheduleFree


v = OptimConfig


class TrainConfig(SweepableConfig):
    data_cfg: DataConfig = Field(default_factory=DataConfig)
    wandb_cfg: dict = Field(default_factory=lambda: dict(project="sae sweeps"))
    coeffs: dict[str, float | Swept[float]] = Field(
        default_factory=lambda: dict(sparsity_loss=1e-3)
    )
    # coeffs: Coeffs = Field(default_factory=Coeffs)
    l0_target: Optional[float] = None
    l0_target_adjustment_size: float = 0.0003
    use_autocast: bool = True
    batch_size: int = 4096
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    use_lars: bool = False
    kwargs: dict = Field(default_factory=dict)
    optim: str = "RAdam"
    raw_schedule_cfg: RunSchedulingConfig = Field(default_factory=RunSchedulingConfig)

    @property
    @lazycall
    def schedule(self):
        return self.raw_schedule_cfg.step_scheduler

    @property
    def use_schedulefree(self):
        return self.optim == "ScheduleFree"

    def get_optim(self):
        return get_optim_cls(self.optim)


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: Trainable,
        namestuff=None,
        optim: torch.optim.Optimizer | None = None,
    ):
        self.cfg: TrainConfig = cfg
        self.model = model
        self.t = 1
        self.log_t_offset = 0
        assert optim is None
        if optim is not None:
            self.optim = optim
        else:
            if self.cfg.use_schedulefree:
                self.optim = AdamWScheduleFree(
                    self.model.parameters(),
                    lr=cfg.lr,
                    betas=cfg.betas,
                    warmup_steps=cfg.schedule.lr_warmup_length,
                )
            else:
                self.optim = get_optim_cls(self.cfg.optim)(
                    self.model.parameters(),
                    lr=cfg.lr,
                    betas=cfg.betas,
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optim, lr_lambda=self.get_lr_lambda()
        )
        if self.cfg.use_lars:
            from torchlars import LARS

            self.optim = LARS(self.optim)
            assert optim is None or not isinstance(optim, LARS)
        # if self.cfg.use_schedulefree:
        # self.optim = ScheduleFreeWrapper(self.optim)

        self.namestuff = namestuff
        self.llm_val_tokens = TokensData(
            self.cfg.data_cfg, self.subject_model
        ).get_tokens_from_split(self.cfg.data_cfg.testsplit)
        self.intermittent_metric_freq = 1000
        self.eval_step_freq = 100
        self.gradscaler = torch.cuda.amp.GradScaler() if self.cfg.use_autocast else None
        self.l0_targeter = L0Targeter(
            l0_target=self.cfg.l0_target,
            schedule=self.cfg.schedule,
        )

    def get_lr_lambda(self):
        if self.cfg.use_schedulefree:

            def lrl(epoch):
                lr_factor = self.cfg.schedule.lr_scale_schedulefree(self.t)
                if self.t % 5 == 0:
                    self.log({"lr_factor": lr_factor})
                return lr_factor

        else:

            def lrl(epoch):
                lr_factor = self.cfg.schedule.lr_scale(self.t)
                if self.t % 5 == 0:
                    self.log({"lr_factor": lr_factor})
                return lr_factor

        return lrl

    @property
    def subject_model(self):
        return self.cfg.data_cfg.model_cfg.model

    def post_backward(self):
        do_post_backward(self.model)

    def post_step(self):
        do_post_step(self.model)

    def log(self, d):
        if wandb.run is not None:
            wandb.log(d, step=self.t + self.log_t_offset)

    def coeffs(self):
        # TODO
        return self.cfg.coeffs

    def proc_cache_after_forward(self, cache: TrainCache):
        if self.cfg.l0_target is not None:

            if not self.cfg.schedule.dynamic_adjust(self.t):
                return
            step = self.l0_targeter(l0=cache.L0, t=self.t)
            stepscale = self.cfg.schedule.targeting_step_scale(self.t)
            self.log(self.l0_targeter.loggables(self.t))
            with torch.no_grad():
                self.cfg.coeffs["sparsity_loss"] = self.cfg.coeffs["sparsity_loss"] * (
                    2 ** (stepscale * step * self.cfg.l0_target_adjustment_size)
                )
            self.log({"dynamic_sparsity_coeff": self.cfg.coeffs["sparsity_loss"]})

    def get_databuffer(self, num_batches=None, num_workers=0):
        return self.cfg.data_cfg.get_databuffer(
            num_workers=num_workers, batch_size=self.cfg.batch_size
        )

    def get_cache(self):
        c = TrainCache()
        c._watch(self.model.get_losses_and_metrics_names())
        c.trainer = ...
        c.trainer = self
        c.trainstep = self.t
        # for k in self.model.get_losses_and_metrics_names():
        # setattr(c, k, ...)
        return c

    def train(self, buffer=None, num_steps=None):
        if buffer is None:
            buffer = self.get_databuffer(num_workers=0)
        if not self.model.normalizer.primed:
            self.model.normalizer.prime_normalizer(buffer)
        self.post_step()
        if wandb.run is None:
            wandb.init(
                **self.cfg.wandb_cfg,
                config={"model": repr(self.model), "cfg": self.cfg},
                reinit=True,
            )
        if self.namestuff is not None:
            lars = "(lars)" if self.cfg.use_lars else ""
            wandb.run.name = f"{lars}{self.namestuff}[{self.cfg.l0_target}]-{wandb.run.name.split('-')[-1]}"

        # if self.cfg.use_schedulefree:
        #     self.optim.train()

        if num_steps is not None:
            old_buffer = buffer
            n = 0

            def buf():
                nonlocal n
                while True:
                    n += 1
                    yield next(old_buffer)
                    if n >= num_steps:
                        break

            buffer = buf()

        for x in buffer:
            x = x.cuda()
            if not self.cfg.use_autocast:
                x = x.float()
            self.optim.zero_grad()

            cache = self.get_cache()

            self.trainstep(x, cache)

            self.full_log(cache)
            self.t += 1
            del cache.forward_reuse_dict
            cache.destroy_children()
            del cache
            if self.t % self.intermittent_metric_freq == 0:
                self.model.eval()
                self.do_intermittent_metrics()
                self.model.train()
            if self.t % self.eval_step_freq == 0:
                self.model.eval()
                self.eval_step(x)
                self.model.train()
            if self.cfg.schedule.is_resample_step(self.t):
                self.model.resampler.resample(
                    data_source=buffer, optimizer=self.optim, model=self.model
                )
                self.post_step()
            if self.cfg.schedule.run_length and self.t > self.cfg.schedule.run_length:
                break

    def trainstep(self, x, cache: Cache):
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
        self.lr_scheduler.step()
        self.post_step()

    def eval_step(self, x):
        cache = self.get_cache()
        if self.cfg.use_autocast:
            with torch.autocast(device_type="cuda"):
                loss = self.model.loss(x, cache=cache, coeffs=self.coeffs())
        else:
            loss = self.model.loss(x, cache=cache, coeffs=self.coeffs())
        self.eval_log(cache)

    def do_intermittent_metrics(self, buffer=None):
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
        if self.t % 10 != 0 and self.t % 23000 > 1000:
            return
        # if wandb.run is not None:
        self.log(
            cache.logdict(exclude_contains=["normalization/mu", "normalization/std"])
        )

    def eval_log(self, cache: Cache):
        d = {
            k: v
            for (k, v) in cache.logdict(name="eval").items()
            if len(k.split("/")) <= 2
        }
        # if wandb.run is not None:
        self.log(d)

    def save(self):
        torch.save(
            self.sae.state_dict(), "/root/workspace/" + wandb.run.name + f"_{self.t}.pt"
        )
