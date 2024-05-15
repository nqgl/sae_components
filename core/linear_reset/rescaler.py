import torch
import torch.nn as nn
from nqgl.mlutils.components.config import WandbDynamicConfig


class RescalerConfig(WandbDynamicConfig): ...


class Rescaler(nn.Module):
    def __init__(self, cfg, module):
        self.module = module
        ...
        self.scaling_factor

    def forward(self, x):
        if self.training:
            self.update_scaling(x)
        return self.unscale(self.module(self.scale(x)))

    # @torch.no_grad()
    def scale(self, x):
        return x / self.scaling_factor

    # @torch.no_grad()
    def unscale(self, x):
        return x * self.scaling_factor

    @torch.no_grad()
    def update_scaling(self, x: torch.Tensor):
        x_cent = x - x.mean(dim=0)
        var = x_cent.norm(dim=-1).pow(2).mean()
        std = torch.sqrt(var)
        self.std_dev_accumulation += (
            std  # x_cent.std(dim=0).mean() is p diferent I believe
        )
        self.std_dev_accumulation_steps += 1
        self.scaling_factor.data[:] = (
            self.std_dev_accumulation
            / self.std_dev_accumulation_steps
            / self.cfg.data_rescale
        )
