import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class L2Normalizer(nn.Module):
    def __init__(self, d_data):
        super().__init__()
        self.d_data = d_data

    def normalize_fn(self, f):
        def normalized_f(x: Float[Tensor, "batch d_data"]):
            adj = torch.linalg.vector_norm(x, dim=-1, ord=2, keepdim=True) / (
                self.d_data**0.5
            )
            return adj * f(x / adj)


def l2normalized(meth):
    def normalized_method(self, x: Float[Tensor, "batch d_data"], **kwargs):
        x = x.float()
        with torch.no_grad():
            adj = x.pow(2).sum(-1).sqrt().unsqueeze(-1) / (self.cfg.d_data**0.5)
        return adj * meth(self, x / adj, **kwargs)

    return normalized_method


def l2mean_normalized(meth):
    def normalized_method(self, x: Float[Tensor, "batch d_data"], **kwargs):
        x = x.float()
        with torch.no_grad():
            adj = x.pow(2).sum(-1).sqrt().mean() / (self.cfg.d_data**0.5)
        return adj * meth(self, x / adj, **kwargs)

    return normalized_method


class L2NormalizerMixin(nn.Module):
    def __init__(self, d_data):
        super().__init__()
        self.d_data = d_data

        self.register_buffer("norm_adjustment", torch.zeros(0))

    def prime(self, buffer, n=10):
        norms = []
        for _ in range(n):
            sample = next(buffer)
            norms.append(torch.linalg.norm(sample, ord=2, dim=-1).mean())
        self.est_norm = torch.tensor(norms).mean() / self.d_data**0.5

    def forward(self, x, **kwargs):
        x_normed = x / self.norm_adjustment
        return super().forward(x_normed, **kwargs) * self.norm_adjustment
