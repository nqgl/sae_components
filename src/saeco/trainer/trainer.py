from contextlib import contextmanager
from functools import cached_property
from typing import Iterable

import torch
import torch.utils
import tqdm

from schedulefree import AdamWScheduleFree
from torch.amp.grad_scaler import GradScaler

from saeco.core import Cache
from saeco.data.tokens_data import TokensData
from saeco.misc.paths import SAVED_MODELS_DIR
from saeco.mlog import mlog
from .call_training_hooks import (
    do_post_backward,
    do_post_forward,
    do_post_step,
    do_pre_forward,
)
from .l0targeter import TARGETER_TYPES
from .OptimConfig import get_optim_cls
from .recons import get_recons_loss
from .run_config import RunConfig
from .train_cache import TrainCache
from .train_config import TrainConfig
from .trainable import Trainable

# torch.multiprocessing.set_start_method("spawn")


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        run_cfg: RunConfig,
        model: Trainable,
        optim: torch.optim.Optimizer | None = None,
        save_callback=None,
    ):
        self.cfg: TrainConfig = cfg
        self.run_cfg: RunConfig = run_cfg
        self.trainable = model
        self.save_callback = save_callback
        self.t = 1
        self.log_t_offset = 0
        self.log_freq = 10

        assert optim is None
        if optim is not None:
            self.optim = optim
        else:
            if self.cfg.use_schedulefree:
                opt_kwargs = dict(
                    lr=cfg.lr,
                    betas=cfg.betas,
                    warmup_steps=cfg.schedule.lr_warmup_length,
                )
                if cfg.weight_decay is not None:
                    opt_kwargs["weight_decay"] = cfg.weight_decay
                self.optim = AdamWScheduleFree(
                    self.trainable.param_groups(opt_kwargs),
                    **opt_kwargs,
                )
            else:
                opt_kwargs = dict(
                    lr=cfg.lr,
                    betas=cfg.betas,
                )
                if cfg.weight_decay is not None:
                    opt_kwargs["weight_decay"] = cfg.weight_decay
                self.optim = get_optim_cls(self.cfg.optim)(
                    self.trainable.param_groups(opt_kwargs),
                    **opt_kwargs,
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optim, lr_lambda=self.get_lr_lambda()
        )
        if self.cfg.use_lars:
            from torchlars import LARS

            self.optim = LARS(self.optim)
            assert optim is None or not isinstance(optim, LARS)

        self.eval_step_freq = 100

        self.gradscaler = GradScaler() if self.cfg.use_autocast else None
        self.l0_targeter = TARGETER_TYPES[self.cfg.l0_targeter_type](
            l0_target=self.get_l0_target(),
            schedule=self.cfg.schedule,
        )
        if self.cfg.use_averaged_model:
            self.averaged_model: torch.optim.swa_utils.AveragedModel = (
                torch.optim.swa_utils.AveragedModel(
                    model,
                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
                )
            )

    @cached_property
    def llm_val_tokens(self):
        return TokensData(
            self.cfg.data_cfg, self.subject_model, split=self.cfg.data_cfg.testsplit
        ).get_tokens()

    def get_l0_target(self):
        if self.cfg.l0_target is None:
            return None

        def target():
            return self.cfg.l0_target * self.cfg.schedule.targeting_multiplier(self.t)

        return target

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

    def pre_forward(self, cache):
        do_pre_forward(self.trainable, cache)

    def post_forward(self, cache):
        do_post_forward(self.trainable, cache)

    def post_backward(self, cache=None):
        do_post_backward(self.trainable, cache)

    def post_step(self, cache=None):
        do_post_step(self.trainable, cache)

    def log(self, d):
        mlog.log(d, step=self.t + self.log_t_offset)

    def coeffs(self):
        self.cfg.schedule
        return self.cfg.coeffs

    @contextmanager
    def evaluate(self):
        self.trainable.eval()
        if self.cfg.use_averaged_model:
            self.averaged_model.eval()
        try:
            yield
        finally:
            self.trainable.train()
            if self.cfg.use_averaged_model:
                self.averaged_model.train()

    def proc_cache_after_forward(self, cache: Cache):
        if self.cfg.l0_targeting_enabled and self.cfg.l0_target is not None:

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

    def make_cache(self):
        cache = TrainCache()
        cache._watch(self.trainable.get_losses_and_metrics_names())
        cache.trainer = ...
        cache.trainer = self
        cache.trainstep = self.t
        return cache

    def train(
        self, buffer: Iterable[torch.Tensor] | None = None, num_steps: int | None = None
    ):
        try:
            self._train(buffer=buffer, num_steps=num_steps)
        finally:
            if self.cfg.save_on_complete:
                try:
                    self.save()
                except Exception as e:
                    print(e)

    def _train(self, buffer=None, num_steps=None):
        if buffer is None:
            buffer = self.cfg.get_databuffer()
        if not self.trainable.normalizer.primed:
            self.trainable.normalizer.prime_normalizer(buffer)
        self.post_step()

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
        for x in tqdm.tqdm(buffer, total=num_steps or self.cfg.schedule.run_length):
            input, target = x.input, x.target
            print(input.shape)
            print(target.shape)
            if not self.cfg.use_autocast:
                input = input.float()  # TODO maybe cast other direction instead
                target = target.float()  # TODO maybe cast other direction instead
            self.optim.zero_grad()
            if self.t % self.eval_step_freq == 0 or (
                self.cfg.schedule.run_length
                and self.t > self.cfg.schedule.run_length - 1000
                and self.t % 5 == 0
            ):
                self.trainable.eval()
                self.eval_step(input, y=target)
                self.trainable.train()

            cache = self.make_cache()
            self.trainstep(input, cache, y=target)
            self.full_log(cache)
            self.t += 1
            cache.destruct()
            if self.cfg.use_averaged_model:
                self.averaged_model.update_parameters(self.trainable)
            if self.t % self.cfg.intermittent_metric_freq == 0:
                self.trainable.eval()
                #                 self.do_intermittent_metrics()
                self.trainable.train()
            if (
                self.cfg.checkpoint_period is not None
                and self.t % self.cfg.checkpoint_period == 0
            ):
                self.save()
            if self.cfg.schedule.is_resample_step(self.t):
                self.trainable.resampler.resample(
                    data_source=buffer, optimizer=self.optim, model=self.trainable
                )
                self.post_step()
            if self.cfg.schedule.run_length and self.t > self.cfg.schedule.run_length:
                break

    @contextmanager
    def _nullcontext(self, yield_value=None):
        yield yield_value

    @property
    def train_autocast_dtype(self) -> torch.dtype:
        dt = self.cfg.data_cfg.model_cfg.acts_cfg.autocast_dtype
        if not dt or dt == torch.float32:
            return torch.float16
        return dt

    def cast(self):
        if self.cfg.use_autocast:
            return torch.autocast(
                device_type="cuda",
                dtype=self.train_autocast_dtype,
            )
        return self._nullcontext()

    def handle_backward(self, loss, cache):
        if self.cfg.use_autocast:
            assert self.gradscaler is not None
            self.gradscaler.scale(loss).backward()
        else:
            loss.backward()
        self.post_backward(cache)

    def handle_step(self, cache):
        if self.cfg.use_autocast:
            assert self.gradscaler is not None
            self.gradscaler.step(self.optim)
            self.gradscaler.update()
        else:
            self.optim.step()
        self.lr_scheduler.step()
        self.post_step(cache)

    def trainstep(self, x, cache: TrainCache, y=None):
        self.pre_forward(cache)
        with self.cast():
            loss = self.trainable.loss(x, cache=cache, y=y, coeffs=self.coeffs())
            self.proc_cache_after_forward(cache)
        self.post_forward(cache)
        self.handle_backward(loss, cache)
        self.handle_step(cache)

    def eval_step(self, x, y=None):
        if self.cfg.use_averaged_model:
            model = self.averaged_model.module
        else:
            model = self.trainable
        cache = self.make_cache()
        with torch.no_grad():
            with self.cast():
                loss = model.loss(x, cache=cache, y=y, coeffs=self.coeffs())
        self.eval_log(cache)
        cache.destruct()

    def do_intermittent_metrics(self, buffer=None):
        self.log_recons("recons/with_bos/", True)
        self.log_recons("recons/no_bos/", False)
        self.log_recons("recons/no_bos2/", False, num_batches=50)

    def log_recons(self, label, proc_bos, num_batches=20):
        self.log(
            {
                (label + k): v
                for k, v in get_recons_loss(
                    self.subject_model,
                    self.trainable,
                    tokens=self.llm_val_tokens,
                    cfg=self.cfg.data_cfg.model_cfg.acts_cfg,
                    bos_processed_with_hook=proc_bos,
                    num_batches=num_batches,
                    cast_fn=self.cast,
                ).items()
            }
        )

    def full_log(self, cache: Cache):
        if self.t % self.log_freq != 0:  # and self.t % 23000 > 100:
            return
        self.log(
            cache.logdict(
                exclude_contains=["normalization/mu", "normalization/std"],
                excluded=["act_metrics_name"],
            )
        )
        # if wandb.run is None and self.t % 25 == 0:

        #     d = cache.logdict(
        #         exclude_contains=["normalization/mu", "normalization/std"],
        #         excluded=["act_metrics_name"],
        #     )
        #     n = ["L2_loss", "L2_aux_loss", "sparsity_loss", "l0", "L0"]
        #     for key in n:
        #         k = f"cache/{key}"
        #         if k in d:
        #             print(f"{k}: {d[k]}")

    def eval_log(self, cache: Cache):
        self.log(
            {
                k: v
                for (k, v) in cache.logdict(name="eval").items()
                if len(k.split("/")) <= 2
            }
        )

    def save(self):
        save_dir = SAVED_MODELS_DIR
        name = mlog.get_run_name()
        sweep_name, run_name = name.split(":")
        savename = save_dir / sweep_name / run_name / str(self.t)

        if self.cfg.use_averaged_model:
            self.save_callback(
                savename,
                save_weights=True,
                averaged_weights=self.averaged_model.state_dict(),
            )
        else:
            self.save_callback(savename, save_weights=True, trainable=self.trainable)
