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
from saeco.core.basic_ops import Add, MatMul, Sub, Mul
from saeco.components.ops.fnlambda import Lambda
from saeco.core.reused_forward import ReuseForward, ReuseCache
from saeco.core import Seq
import saeco.components.features.features as ft
import saeco.components as co
from saeco.trainer.trainable import Trainable


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


class Tied:
    INIT = 0
    TIED = 1

    def __init__(self, target: "LinearFactory", tie_type, site: str):
        self.target = target
        self.tie_type = tie_type
        self.site = site

    def __call__(self, other: nn.Linear):
        src_param = getattr(self.target.raw, self.site)
        if self.tie_type == self.INIT:
            dst_param = getattr(other, self.site)
            assert (dst_param.data.shape == src_param.data.shape) ^ (
                dst_param.data.shape == src_param.data.transpose(-2, -1).shape
            )
            if dst_param.data.shape == src_param.data.shape:
                dst_param.data[:] = src_param.data
            else:
                dst_param.data[:] = src_param.data.transpose(-2, -1)
        elif self.tie_type == self.TIED:
            setattr(other, self.site, src_param)
        else:
            raise ValueError("Invalid tie type")


class LinearFactory:
    def __init__(self, d_in, d_out, bias=True, wrappers=[]):
        self.d_in = d_in
        self.d_out = d_out
        self._bias = bias
        self._linear = None
        self._linear_raw = None
        self.wrappers = wrappers
        self._weight_tie: Optional[Tied] = None
        self._bias_tie: Optional[Tied] = None

    @property
    def unset(self):
        l = self._linear is None
        lr = self._linear_raw is None
        assert l == lr
        return l

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        if not self.unset:
            raise ValueError("Cannot change bias after linear has been created")
        self._bias = value

    def add_wrapper(self, wrapper):
        if not self.unset:
            raise ValueError("Cannot add wrappers after linear has been created")
        self.wrappers.append(wrapper)

    def make_new(self):
        lin = nn.Linear(self.d_in, self.d_out, bias=self.bias)
        if self._weight_tie is not None:
            self._weight_tie(lin)
        if self._bias_tie is not None:
            self._bias_tie(lin)
        self._linear_raw = lin
        for w in self.wrappers:
            lin = w(lin)
        return lin

    def get(self) -> nn.Linear:
        if self._linear is None:
            self._linear = self.make_new()
        return self._linear

    @property
    def raw(self):
        if self._linear_raw is None:
            self.get()
        return self._linear_raw

    @property
    def lin(self):
        return self.get()

    @property
    def detached(self):
        return DetachedLinear(self.lin)  # should this use raw?

    def tie_weights(self, other):
        assert self.unset
        # self.lin.weight = other.lin.weight
        assert self._weight_tie is None
        self._weight_tie = Tied(other, Tied.TIED, "weight")

    def tied_weights_init(self, other):
        assert self.unset
        assert self._weight_tie is None
        self._weight_tie = Tied(other, Tied.INIT, "weight")

        # assert (self.lin.weight.data.shape == other.lin.weight.data.shape) ^ (
        #     self.lin.weight.data.shape == other.lin.weight.data.transpose(-2, -1).shape
        # )
        # if self.lin.weight.data.shape == other.lin.weight.data.shape:
        #     self.lin.weight.data[:] = other.lin.weight.data
        # else:
        #     self.lin.weight.data[:] = other.lin.weight.data.transpose(-2, -1)

    def sub_bias(self) -> Sub:
        return Sub(self.lin.bias)


class DetachedLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        return torch.nn.functional.linear(
            x,
            self.lin.weight.detach(),
            self.lin.bias.detach() if self.lin.bias is not None else None,
        )


class Initializer:
    def __init__(self, d_data, d_dict=None, dict_mult=None):
        self.d_data = d_data
        d_dict = d_dict or d_data * dict_mult
        self.d_dict = d_dict
        # self.tied_bias = True
        self.tied_init = True
        self.tied_weights = False
        self.encoder_init_weights = None

        self._decoder: LinearFactory = LinearFactory(
            d_dict, d_data, wrappers=[co.LinDecoder]
        )
        self._encoder: LinearFactory = LinearFactory(
            d_data, d_dict, wrappers=[co.LinEncoder]
        )
        # self._encoder.weight.data[:] = self._encoder.weight * 3**0.5
        # self._decoder.weight.data[:] = self._encoder.weight.transpose(-2, -1)
        if self.tied_init:
            self._decoder.tied_weights_init(self._encoder)
        if self.tied_weights:
            self._decoder.tie_weights(self._encoder)

    @property
    def encoder(self):
        return self._encoder.get()

    @property
    def decoder(self):
        return self._decoder.get()

    @property
    def b_dec(self):
        return self._decoder.get().bias

    def bias(self, d, name=None):
        return nn.Parameter(torch.zeros(d))

    def dict_bias(self):
        return self.bias(self.d_dict)

    def new_encoder_bias(self):
        return co.EncoderBias(Add(self.bias(self.d_dict)))

    def data_bias(self):
        return self.bias(self.d_data)

    # def get_bias(self,):
    # )
