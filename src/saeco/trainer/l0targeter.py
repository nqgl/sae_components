import torch
from typing import Protocol


class L0Targeter:
    target: float
    lr: float

    def __init__(self, l0_target: float, l0_target_adjustment_size: float):
        self.target = l0_target
        self.lr = l0_target_adjustment_size

    def __call__(self, l0):
        stepscale = 1
        period = (
            self.cfg.schedule.resample_dynamic_cooldown_period_override
            or self.cfg.schedule.resample_period
        )

        if self.t % period < self.cfg.schedule.resample_dynamic_stop_after:
            return

        if self.t < period + self.cfg.schedule.resample_delay:
            stepscale *= min(
                1,
                ((self.t % period) - self.cfg.schedule.resample_dynamic_stop_after)
                / (
                    (period - self.cfg.schedule.resample_dynamic_stop_after)
                    * self.cfg.schedule.resample_dynamic_cooldown
                ),
            )

        gentle_zone_radius = 5
        distance = abs(cache.L0 - self.cfg.l0_target) / gentle_zone_radius
        stepscale *= min(
            1,
            (distance**0.5 * 4 + distance * 2 + 1) / 7,
        )

        # if self.t % self.cfg.resample_freq < self.cfg.resample_freq * self.cfg.resample_dynamic_cooldown:
        self.cfg.coeffs["sparsity_loss"] = self.cfg.coeffs["sparsity_loss"] * (
            1
            + (-1 if self.cfg.l0_target > cache.L0 else 1)
            * self.cfg.l0_target_adjustment_size
            * stepscale
        )
