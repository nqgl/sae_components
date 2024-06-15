import torch
from .penalty import Penalty
from torch import Tensor


class L1Penalty(Penalty):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def penalty(self, x: Tensor):
        return x.abs().mean(dim=0).sum() * self.scale
