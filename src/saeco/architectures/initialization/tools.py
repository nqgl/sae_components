import torch
import torch.nn as nn

from saeco.core import Seq
from saeco.core.basic_ops import Add, MatMul
from saeco.core.reused_forward import ReuseForward

from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float
from typing import Optional

from saeco.components.ops.detach import Thresh
import saeco.core as cl
import saeco.core.module
from saeco.core.collections.parallel import Parallel
from saeco.components import (
    Penalty,
    L1Penalty,
    FreqTracked,
    EMAFreqTracker,
    FreqTracker,
    ResampledLayer,
    Loss,
    L2Loss,
    SparsityPenaltyLoss,
    SAECache,
)

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Add, MatMul, Mul
from saeco.components.ops.fnlambda import Lambda
from saeco.core.reused_forward import ReuseForward, ReuseCache
from saeco.core import Seq
import saeco.components.features.features as ft


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


def layer(d_in, d_out, nonlinearity=nn.LeakyReLU, scale=1.0):
    if nonlinearity is nn.PReLU:
        nonlinearity = nn.PReLU(d_out).cuda()

    if isinstance(nonlinearity, type):
        nonlinearity = nonlinearity()
    return Seq(
        weight=MatMul(weight(d_in, d_out, scale=scale)),
        bias=Add(bias(d_out)),
        nonlinearity=nonlinearity,
    )


def mlp_layer(d_in, d_hidden, d_out=None, nonlinearity=nn.LeakyReLU, scale=1.0):
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
