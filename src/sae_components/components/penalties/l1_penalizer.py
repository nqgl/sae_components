import torch
from sae_components.components.penalties.penalty import Penalty
from torch import Tensor


class L1Penalty(Penalty):
    def penalty(self, x: Tensor):
        return x.abs().mean(dim=0).sum()
