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


class ReMaxK(nn.Module):
    def __init__(self, k, scale=1, norm=1, b=False):
        super().__init__()
        self.scale = scale
        assert norm in (1, 2)
        self.norm = norm
        self.k = k
        self.b = b

    def forward(self, x):
        v = torch.relu(x)
        v, i = x.topk(self.k, dim=-1, sorted=False)
        vk = torch.zeros_like(x).scatter_(-1, i, v)

        mag = (
            v.sum(dim=-1, keepdim=True)
            if self.norm == 1
            else v.norm(dim=-1, keepdim=True)
        )
        magk = (
            vk.sum(dim=-1, keepdim=True)
            if self.norm == 1
            else v.norm(dim=-1, keepdim=True)
        )
        if self.b:
            out = vk / (1 + mag - magk)
        else:
            # out = vk / mag * magk
            out = vk / (mag - magk / (2**0.5) + 1e-7)

        return torch.where(torch.isnan(out) | torch.isinf(out), 0, out) * self.scale


class ReMaxKv(nn.Module):
    def __init__(self, k, scale=1, norm=1, b=False):
        super().__init__()
        self.scale = scale
        assert norm in (1, 2)
        self.norm = norm
        self.k = k
        self.b = b

    def forward(self, x):
        v = torch.relu(x)
        val, i = x.topk(self.k, dim=-1, sorted=False)
        vk = torch.zeros_like(x).scatter_(-1, i, val)

        mag = (
            v.sum(dim=-1, keepdim=True)
            if self.norm == 1
            else v.norm(dim=-1, keepdim=True)
        )
        magk = (
            vk.sum(dim=-1, keepdim=True)
            if self.norm == 1
            else v.norm(dim=-1, keepdim=True)
        )
        if self.b:
            out = v / (mag - magk + 1)
        else:
            out = v / mag * magk
        return torch.where(torch.isnan(out) | torch.isinf(out), 0, out) * self.scale
