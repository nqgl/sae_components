import torch
import torch.nn as nn


class ReMax(nn.Module):
    def __init__(self, scale=1, norm=1):
        super().__init__()
        self.scale = scale
        assert norm in (1, 2)
        self.norm = norm

    def forward(self, x):
        v = torch.relu(x)
        mag = (
            v.sum(dim=-1, keepdim=True)
            if self.norm == 1
            else v.norm(dim=-1, keepdim=True)
        )
        out = v / mag
        return torch.where(torch.isnan(out) | torch.isinf(out), 0, out) * self.scale


class ReMax1(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        v = torch.relu(x)
        out = v / (v.sum(dim=-1, keepdim=True) - v)
        return torch.where(torch.isnan(out) | torch.isinf(out), 0, out) * self.scale
