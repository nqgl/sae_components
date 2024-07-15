import torch
from typing import Protocol, Optional
from .schedule_cfg import RunSchedulingConfig


class L0TargeterProto(Protocol):
    target: float
    schedule: RunSchedulingConfig

    def __call__(self, l0: float, t: int) -> float: ...


class L0Targeter(L0TargeterProto):
    def __init__(
        self,
        l0_target: Optional[float],
        schedule: RunSchedulingConfig,
    ):
        self.target = l0_target
        self.schedule = schedule

    def __call__(self, l0: float, t: int) -> float:
        if self.target is None:
            return 0
        if not self.schedule.dynamic_adjust(t):
            return 0
        stepscale = 1
        # period = (
        #     self.schedule.targeting_resample_cooldown_period_override
        #     or self.schedule.resample_period
        # )
        rt = self.schedule.resample_t(t)
        t_after_wait = rt - self.schedule.targeting_post_resample_hiatus
        res_warmup_len = (
            self.schedule.targeting_post_resample_cooldown
            - self.schedule.targeting_post_resample_hiatus
        )
        if t_after_wait >= 0:
            stepscale *= min(1, t_after_wait / res_warmup_len)

        gentle_zone_radius = 5
        distance = abs(l0 - self.target) / gentle_zone_radius
        stepscale *= min(
            1,
            (distance * 6 + 1) / 7,
        )

        return (-1 if self.target > l0 else 1) * stepscale
