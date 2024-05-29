import torch
import torch.nn as nn


class ReMax(nn.Module):
    def forward(self, x):
        v = torch.relu(x)
        out = v / v.sum(dim=-1, keepdim=True)
        return torch.where(torch.isnan(out) | torch.isinf(out), 0, out)


class ReMax1(nn.Module):
    def forward(self, x):
        v = torch.relu(x)
        out = v / (v.sum(dim=-1, keepdim=True) - v)
        return torch.where(torch.isnan(out) | torch.isinf(out), 0, out)
