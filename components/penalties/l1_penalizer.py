import torch
from components.sparsity_penalizers.penalty import Penalty
from torch import Tensor
from sae_components.core.collection_modules.seq import SequentialCacheLayer


class L1Penalty(Penalty):
    def penalty(self, x: Tensor):
        return x.abs().mean(dim=0).sum()
