from collections.abc import Callable

import torch
import torch.nn as nn

from saeco.core import Seq
from saeco.core.basic_ops import Add, MatMul
from saeco.core.module import Module
from saeco.core.reused_forward import ReuseForward


def bias(d: int) -> nn.Parameter:
    return nn.Parameter(torch.zeros(d))


def weight(
    d1: int, d2: int, scale: float = 1.0, transpose: bool = False
) -> nn.Parameter:
    if transpose:
        return nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(d2, d1)).transpose(-1, -2) * scale
        )
    return nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d1, d2)) * scale)


def reused[**P, T: Module](func: Callable[P, T]) -> Callable[P, ReuseForward[T]]:
    v = None

    def f(*a: P.args, **k: P.kwargs) -> ReuseForward[T]:
        nonlocal v
        if v is None:
            v = ReuseForward(func(*a, **k))
        return v

    return f


def layer(
    d_in: int,
    d_out: int,
    nonlinearity: type[nn.Module] | nn.Module = nn.LeakyReLU,
    scale: float = 1.0,
) -> Seq:
    if nonlinearity is nn.PReLU:
        nonlinearity = nn.PReLU(d_out).cuda()

    if isinstance(nonlinearity, type):
        nonlinearity = nonlinearity()
    return Seq(
        weight=MatMul(weight(d_in, d_out, scale=scale)),
        bias=Add(bias(d_out)),
        nonlinearity=nonlinearity,
    )


def mlp_layer(
    d_in: int,
    d_hidden: int,
    d_out: int | None = None,
    nonlinearity: type[nn.Module] | nn.Module = nn.LeakyReLU,
    scale: float = 1.0,
) -> Seq:
    d_out = d_out or d_in
    if nonlinearity is nn.PReLU:
        nonlinearity = nn.PReLU(d_hidden).cuda()
    if isinstance(nonlinearity, type):
        nonlinearity = nonlinearity()

    return Seq(
        weight=MatMul(weight(d_in, d_hidden, scale=scale)),
        h_bias=Add(bias(d_hidden)),
        nonlinearity=nonlinearity,
        projection=MatMul(weight(d_hidden, d_out, scale=scale)),
        o_bias=Add(bias(d_out)),
    )
