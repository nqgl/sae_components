import torch


def rect(x: torch.Tensor) -> torch.Tensor:
    return x.abs() < 0.5


def tri(x: torch.Tensor) -> torch.Tensor:
    return (1 - x.abs()).clamp(0, 1)


def exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-x.abs()) / 2 / torch.e


kernels = dict(rect=rect, tri=tri, exp=exp)
