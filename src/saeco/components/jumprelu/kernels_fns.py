import torch
from typing import Protocol


def rect(x: torch.Tensor) -> torch.Tensor:
    return x.abs() < 0.5


def tri(x: torch.Tensor) -> torch.Tensor:
    return (1 - x.abs()).clamp(0, 1)


def exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-x.abs() * 2)


def trap(x: torch.Tensor) -> torch.Tensor:
    return (1 - x.abs() * 3 / 2).clamp(0, 0.5) * 2


def gauss(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-((x) ** 2) * torch.pi)


def bimodal(x: torch.Tensor) -> torch.Tensor:
    # would expect this to be bad, just for a sanity check
    x = x * 3
    return (x**2 + (x.abs() + 1).pow(0.5) / 2 - x.abs() / 4) * (-x.pow(4)).exp() * 2


def softtrap(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-((x * 2.16) ** 4) / 2)


kernels = dict(
    rect=rect,
    tri=tri,
    exp=exp,
    trap=trap,
    gauss=gauss,
    bimodal=bimodal,
    softtrap=softtrap,
)


class Kernel(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


def integrate(kernel: Kernel, range=10, samples=8192):
    x = torch.linspace(-range, range, samples)
    y = kernel(x)
    return y.sum() / samples * 2 * range


def centrality(kernel, centrality_measure=gauss):
    f = lambda x: centrality_measure(x) * kernel(x)
    return integrate(f)


def check(kernel: Kernel, range=10, samples=8192):
    f2 = lambda x: x * kernel(x)
    f3 = lambda x: x**2 * kernel(x)
    # print(f"kernel.__name__)
    v1 = integrate(kernel, range, samples)
    v2 = integrate(f2, range, samples).abs()
    v3 = integrate(f3, range, samples)
    l = 10 - len(kernel.__name__)
    print(
        f"{kernel.__name__}:{' ' * l} {v1:.2f}, {v2:.2f}, {v3:.2f}, [ {kernel(torch.zeros(1)).item():.2f}, cent: {centrality(kernel):.2f} ]"
    )


if __name__ == "__main__":
    check(rect)
    check(tri)
    check(exp)
    check(trap)
    check(gauss)
    check(bimodal)
    check(softtrap)
