# %%
from saeco.sweeps import SweepableConfig, Swept
from typing import Any, Optional, overload, Callable
from functools import wraps
from pydantic import Field


AmbiguousTypes = [Optional[int | float], int | float]
# Run length
# ", resample period
from saeco.trainer.tosteps_wrapper import RunFloat, ResFloat, tosteps_wrapper


def assert_wrapped(fn):
    assert not isinstance(fn, property)

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        assert getattr(
            self.__class__, "_IS_WRAPPED", False
        ), "Cannot call methods on the raw schedule. Access the wrapped object via .step_scheduler instead."
        return fn(self, *args, **kwargs)

    return wrapper


class RunSchedulingConfig(SweepableConfig):
    run_length: Optional[int] = 50_000

    resample_period: int = 12_500
    resample_delay: int | RunFloat = 0
    resampling_finished_phase: int | RunFloat = 0.3

    targeting_post_resample_cooldown: int | ResFloat = 0.2
    # targeting_resample_cooldown_period_override: Optional[int] = None
    targeting_post_resample_hiatus: int | ResFloat = 0.05
    targeting_delay: int | RunFloat = 0  # could be none -> copy cooldown
    targeting_warmup_length: int | RunFloat = 0.15

    ### lr scheduler # this is not quite the continuous pretraining scheduler, seems fine though
    lr_warmup_length: int | RunFloat = 0.05
    lr_cooldown_length: int | RunFloat = 0.2
    lr_resample_warmup_length: int | ResFloat = 0.2
    lr_warmup_factor: float = 0.0
    lr_cooldown_factor: float = 0.0
    lr_resample_warmup_factor: float = 0.0

    # def model_post_init(self):
    @property
    def step_scheduler(self) -> "RunSchedulingConfig":
        return tosteps_wrapper(self.__class__)(_raw=self)

    @assert_wrapped
    def targeting_step_scale(self, t):
        stepscale = 1
        res_t_since_hiatus = self.resample_t(t) - self.targeting_post_resample_hiatus
        if res_t_since_hiatus >= 0:
            stepscale *= min(
                1, res_t_since_hiatus / self.targeting_post_resample_cooldown
            )
        if t < self.targeting_warmup_length:
            stepscale *= t / self.targeting_warmup_length
        return stepscale

    # @property
    # @assert_wrapped
    # def resample_start(self) -> int:
    #     return self.resample_delay

    @property
    @assert_wrapped
    def resample_end(self) -> int:
        return self.run_length - self.resampling_finished_phase

    @assert_wrapped
    def dynamic_adjust(self, t):
        if t < self.targeting_delay:
            return False
        rt = self.resample_t(t)
        if rt != -1 and rt < self.tosteps(
            self.targeting_post_resample_hiatus, self.resample_period
        ):
            return False
        return True

    @assert_wrapped
    def tosteps(self, n: int | float, period: int = None) -> int:
        # some values will be expressed as either
        # a number of steps
        # or a fraction of some period, default run length
        # signified by type -- ints are steps, floats are proportions
        # this converts proportions to steps and leaves steps as is
        assert 0 <= n
        if isinstance(n, int):
            return n
        assert isinstance(n, float) and n <= 1
        period = period or self.run_length
        return n * period

    @assert_wrapped
    def lr_scale(self, t: int) -> float:
        re_lr = 1
        if self.lr_resample_warmup_length and (rt := self.resample_t(t)) != -1:
            re_warmup = self.tosteps(
                self.lr_resample_warmup_length, self.resample_period
            )
            re_lr = max(min(rt / re_warmup, 1), self.lr_resample_warmup_factor)
        warmup = self.tosteps(self.lr_warmup_length)
        if t < warmup:
            return re_lr * max(t / warmup, self.lr_warmup_factor)
        to_end = self.run_length - t
        cooldown = self.tosteps(self.lr_cooldown_length)
        if to_end < cooldown:
            return re_lr * max(to_end / cooldown, self.lr_cooldown_factor)
        return re_lr

    @assert_wrapped
    def lr_scale_schedulefree(self, t: int) -> float:
        re_lr = 1
        if self.lr_resample_warmup_length and (rt := self.resample_t(t)) != -1:
            re_warmup = self.tosteps(
                self.lr_resample_warmup_length, self.resample_period
            )
            re_lr = max(min(rt / re_warmup, 1), self.lr_resample_warmup_factor)
        return re_lr

    @assert_wrapped
    def resample_t(self, t: int) -> int:
        if t < self.resample_delay:
            return -1
        if self.resample_delay == 0 and t < self.resample_period:
            return -1
        if t - self.resample_delay + self.resample_period > self.resample_end:
            return -1
        return (t - self.resample_delay) % self.resample_period

    @assert_wrapped
    def is_resample_step(self, t: int) -> bool:
        return self.resample_t(t) == 0
        if t < self.resample_delay:
            return False
        if (t - self.resample_delay) % self.resample_period == 0:
            return True
        return False
