from saeco.sweeps import SweepableConfig
from typing import Optional, overload, Callable
from functools import wraps


@overload
def tosteps(n: int, period: None) -> Callable[[int | float], int]: ...


@overload
def tosteps(n: int | float, period: int) -> int: ...


def tosteps(n: int | float, period: int | None = None) -> str:
    # some values will be expressed as either
    # a number of steps
    # or a fraction of some period, default run length
    # signified by type -- ints are steps, floats are proportions
    # this converts proportions to steps and leaves steps as is
    assert 0 <= n
    if period is None:
        period = n

        @wraps(tosteps)
        def inner(n: int | float) -> int:
            return tosteps(n, period)

        return inner
    if isinstance(n, int):
        return n
    assert isinstance(n, float) and n <= 1 and isinstance(period, int)
    return n // period


class RunSchedulingConfig(SweepableConfig):
    run_length: Optional[int] = 5e3

    resample_period: int = 2_000
    resample_delay: int = 0
    resampling_finished_phase: int | float = 0

    resample_dynamic_cooldown: float = 0.1
    resample_dynamic_cooldown_period_override: Optional[int] = None
    resample_dynamic_stop_after: int = 0

    ### lr scheduler # this is not quite the continuous pretraining scheduler, seems fine though
    lr_warmup_length: int | float = 0.1
    lr_cooldown_length: int | float = 0.2
    lr_resample_warmup_length: Optional[int | float] = 150
    lr_warmup_factor: float = 0.1
    lr_cooldown_factor: float = 0.1
    lr_resample_warmup_factor: float = 0.1

    @property
    def resample_start(self) -> int:
        return self.tosteps(self.resample_delay)

    @property
    def resample_end(self) -> int:
        return self.run_length - self.tosteps(self.resampling_finished_phase)

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

    def resample_t(self, t: int) -> int:
        if t < self.resample_delay:
            return -1
        if t - self.resample_delay + self.resample_period > self.resample_end:
            return -1
        return (t - self.resample_delay) % self.resample_period

    def is_resample_step(self, t: int) -> bool:
        return self.resample_t(t) == 0
        if t < self.resample_delay:
            return False
        if (t - self.resample_delay) % self.resample_period == 0:
            return True
        return False


rs = RunSchedulingConfig()
print(type(rs.resampling_finished_phase))
