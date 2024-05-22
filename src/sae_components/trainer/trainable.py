import sae_components.core.module as cl
from sae_components.core import Cache
import torch
import wandb
from typing import Protocol, runtime_checkable, List
import torch.nn as nn
from sae_components.components.losses import Loss, L2Loss, SparsityPenaltyLoss
from dataclasses import dataclass


class Trainable(cl.Module):
    losses: List[Loss]
    models: nn.ModuleList[cl.Module]

    def forward(self, x: torch.Tensor, cache: Cache = None) -> torch.Tensor:
        return self.models[0](x, cache)

    def loss(self, x: torch.Tensor, cache: Cache = None) -> torch.Tensor:
        return torch.sum(loss(x, cache) for loss in self.losses)
