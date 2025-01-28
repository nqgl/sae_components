from contextlib import contextmanager
from typing import Optional, Protocol, runtime_checkable

import torch
import torch.utils
import tqdm
import wandb
from schedulefree import AdamWScheduleFree, ScheduleFreeWrapper
from torch.amp import GradScaler

from saeco.components.losses import L2Loss, SparsityPenaltyLoss
from saeco.core import Cache
from saeco.data.model_cfg import ModelConfig
from saeco.data.tokens_data import TokensData
from .OptimConfig import get_optim_cls, OptimConfig
from .post_backward_normalization import do_post_backward, do_post_step
from .train_cache import TrainCache
from .train_config import TrainConfig
from .trainable import Trainable
from .l0targeter import L0Targeter, TARGETER_TYPES
from .recons import get_recons_loss
from .run_config import RunConfig
from .saved_model_source_info import ModelReloadInfo
from saeco.mlog import mlog

# torch.multiprocessing.set_start_method("spawn")


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        run_cfg: RunConfig,
        model: Trainable,
        run_name=None,
        optim: torch.optim.Optimizer | None = None,
        save_callback=None,
        **kwargs,
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

        self.namestuff = run_name
        self.llm_val_tokens = TokensData(
            self.cfg.data_cfg, self.subject_model, split=self.cfg.data_cfg.testsplit
        ).get_tokens()
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

    def post_backward(self):
        do_post_backward(self.trainable)

    def post_step(self):
        do_post_step(self.trainable)

    def log(self, d):
        # if wandb.run is not None:
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

    def proc_cache_after_forward(self, cache: TrainCache):
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

    def get_databuffer(self, num_batches=None, num_workers=0, queue_size=...):
        buf = self.cfg.data_cfg.get_databuffer(
            num_workers=num_workers, batch_size=self.cfg.batch_size
        )
        if queue_size is ...:
            queue_size = int(4096 / self.cfg.batch_size * 64 + 4)
        if queue_size is not None:
            queue = [next(buf).cuda(non_blocking=True) for _ in range(queue_size)]

            def qbuf():
                try:
                    while True:
                        yield queue.pop(0)
                        queue.append(next(buf).cuda(non_blocking=True))
                except StopIteration:
                    print("GPU buffer source depleted")
                    for x in queue:
                        yield x

            return qbuf()
        return buf

    def get_cache(self):
        c = TrainCache()
        c._watch(self.trainable.get_losses_and_metrics_names())
        c.trainer = ...
        c.trainer = self
        c.trainstep = self.t
        # for k in self.model.get_losses_and_metrics_names():
        # setattr(c, k, ...)
        return c

    def train(self, buffer=None, num_steps=None):
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
            buffer = self.get_databuffer(num_workers=6)
        if not self.trainable.normalizer.primed:
            self.trainable.normalizer.prime_normalizer(buffer)
        self.post_step()
        # if wandb.run is None:
        #     wandb.init(
        #         **self.cfg.wandb_cfg,
        #         config=self.run_cfg.model_dump(),
        #         reinit=True,
        #     )
        if wandb.run is not None:
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

        for x in tqdm.tqdm(buffer, total=num_steps or self.cfg.schedule.run_length):
            if not self.cfg.use_autocast:
                x = x.float()  # TODO maybe cast other direction instead
            self.optim.zero_grad()
            if self.t % self.eval_step_freq == 0 or (
                self.t > self.cfg.schedule.run_length - 1000 and self.t % 5 == 0
            ):
                self.trainable.eval()
                self.eval_step(x)
                self.trainable.train()

            cache = self.get_cache()
            self.trainstep(x, cache)

            self.full_log(cache)
            self.t += 1
            cache.destruct()
            if self.cfg.use_averaged_model:
                self.averaged_model.update_parameters(self.trainable)
            if self.t % self.cfg.intermittent_metric_freq == 0:
                self.trainable.eval()
                self.do_intermittent_metrics()
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

    def cast(self):
        if self.cfg.use_autocast:
            return torch.autocast(
                device_type="cuda",
                dtype=self.cfg.data_cfg.model_cfg.acts_cfg.autocast_dtype,
            )
        return self._nullcontext()

    def handle_backward(self, loss):
        if self.cfg.use_autocast:
            self.gradscaler.scale(loss).backward()
        else:
            loss.backward()
        self.post_backward()

    def handle_step(self):
        if self.cfg.use_autocast:
            self.gradscaler.step(self.optim)
            self.gradscaler.update()
        else:
            self.optim.step()
        self.lr_scheduler.step()
        self.post_step()

    def trainstep(self, x, cache: Cache):
        with self.cast():
            loss = self.trainable.loss(x, cache=cache, coeffs=self.coeffs())
            self.proc_cache_after_forward(cache)
        self.handle_backward(loss)
        self.handle_step()

    def eval_step(self, x):
        if self.cfg.use_averaged_model:
            model = self.averaged_model.module
        else:
            model = self.trainable
        cache = self.get_cache()
        with torch.no_grad():
            with self.cast():
                loss = model.loss(x, cache=cache, coeffs=self.coeffs())
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
        if wandb.run is None and self.t % 25 == 0:

            d = cache.logdict(
                exclude_contains=["normalization/mu", "normalization/std"],
                excluded=["act_metrics_name"],
            )
            n = ["L2_loss", "L2_aux_loss", "sparsity_loss", "l0", "L0"]
            for key in n:
                k = f"cache/{key}"
                if k in d:
                    print(f"{k}: {d[k]}")

    def eval_log(self, cache: Cache):
        d = {
            k: v
            for (k, v) in cache.logdict(name="eval").items()
            if len(k.split("/")) <= 2
        }
        # if wandb.run is not None:
        self.log(d)

    def save(self):
        from pathlib import Path

        save_dir = Path.home() / "workspace/saved_models/"
        if wandb.run:
            if wandb.run.project:
                save_dir /= wandb.run.project
            if wandb.run.sweep_id:
                save_dir /= f"sweep{wandb.run.sweep_id}"
            name = f"{wandb.run.name}/{self.t}"
        else:
            name = f"{self.namestuff}/{self.t}"
        savename = save_dir / name

        if self.cfg.use_averaged_model:
            self.save_callback(
                savename,
                save_weights=True,
                averaged_weights=self.averaged_model.state_dict(),
            )
        else:
            self.save_callback(savename, save_weights=True, trainable=self.trainable)
        # cfg_path = savename.with_suffix(".json")
        # model_path = savename.with_suffix(".pt")
        # reload_info = savename.with_suffix(".reload_info.json")
        # assert (not cfg_path.exists()) and (not model_path.exists())
        # cfg_path.parent.mkdir(exist_ok=True, parents=True)
        # if averaged:
        #     torch.save(self.averaged_model.state_dict(), model_path)
        # else:
        #     torch.save(self.trainable.state_dict(), model_path)
        # cfg_path.write_text(self.run_cfg.model_dump_json())
        # reload_info.write_text(self.reload_info.model_dump_json())
