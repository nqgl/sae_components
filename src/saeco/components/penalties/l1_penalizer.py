import torch
from .penalty import Penalty
from torch import Tensor


class L1Penalty(Penalty):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def penalty(self, x: Tensor):
        return x.relu().mean(dim=0).sum() * self.scale


class L0TargetingL1Penalty(Penalty):
    def __init__(self, target, scale=1.0):
        super().__init__()
        self.scale = scale
        self.target = target
        self.increment = 0.00003

    def penalty(self, x: Tensor):
        return x.relu().mean(dim=0).sum() * self.scale

    def update_l0(self, x: Tensor):
        if self.target is None:
            return x
        l0 = (x > 0).sum(dim=-1).float().mean(0).sum()
        if l0 > self.target:
            self.scale *= 1 + self.increment
        else:
            self.scale *= 1 - self.increment
        return x
