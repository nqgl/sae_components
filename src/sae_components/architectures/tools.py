import torch
import torch.nn as nn

from sae_components.core import Seq
from sae_components.core.basic_ops import Add, MatMul
from sae_components.core.reused_forward import ReuseForward


def bias(d):
    return nn.Parameter(torch.zeros(d))


def weight(d1, d2, scale=1.0, transpose=False):
    if transpose:
        return nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(d2, d1)).transpose(-1, -2) * scale
        )
    return nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d1, d2)) * scale)


def reused(func):
    v = None

    def f(*a, **k):
        nonlocal v
        if v is None:
            v = ReuseForward(func(*a, **k))
        return v

    return f


def layer(d_in, d_out, nonlinearity=nn.LeakyReLU):
    return Seq(
        weight=(MatMul(weight(d_in, d_out))),
        bias=Add(bias(d_out)),
        nonlinearity=nonlinearity(),
    )


def mlp_layer(d_in, d_hidden, d_out=None, nonlinearity=nn.GELU, scale=1.0):
    d_out = d_out or d_in
    return Seq(
        weight=MatMul(weight(d_in, d_hidden, scale=scale)),
        bias=Add(bias(d_hidden)),
        nonlinearity=nonlinearity(),
        projection=MatMul(weight(d_hidden, d_out, scale=scale)),
    )
