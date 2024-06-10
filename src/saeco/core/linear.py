import saeco.core.module
from saeco.core.cache import Cache
import saeco.core as cl

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass
from typing import Tuple, Any, Union, Optional, List
import einops
from unpythonic import box
from abc import abstractmethod, ABC


class FromDims:  # could add some init logic here or make a generalized "parameter initialization" class
    def __init__(self, dims: List[int]):
        self.dims = dims
        self.initalization = nn.init.kaiming_uniform_

    def parameter(self) -> nn.Parameter:
        return nn.Parameter(torch.zeros(*self.dims))


class FromDimsTied(FromDims):
    def __init__(self, dims: List[int]):
        super().__init__(dims)
        self._parameter = None

    def parameter(self) -> nn.Parameter:
        if self._parameter is None:
            self._parameter = nn.Parameter(torch.ones(*self.dims))
        return self._parameter


class Bias(saeco.core.module.Module):
    def __init__(self, bias):
        super().__init__()
        assert isinstance(bias, nn.Parameter)
        self.bias = bias

    @property
    def in_features(self):
        return self.bias.shape[-1]

    @property
    def out_features(self):
        return self.bias.shape[-1]

    def forward(self, x, *, cache: Cache = None, **kwargs):
        return x + self.bias

    def tied_negative(self):
        return NegBias(self.bias)

    def __neg__(self):
        return self.tied_negative()


class NegBias(Bias):
    def forward(self, x, *, cache: Cache, **kwargs):
        return x - self.bias

    def tied_negative(self):
        print("warning: negated a NegBias")
        return Bias(self.bias)


class MatMul(saeco.core.module.Module):
    def __init__(self, W):
        super().__init__()
        assert isinstance(W, nn.Parameter)
        self.weight = W

    @property
    def in_features(self):
        return self.weight.shape[-2]

    @property
    def out_features(self):
        return self.weight.shape[-1]

    def forward(self, x, *, cache: Cache, **kwargs):
        return x @ self.weight


class Affine(cl.Seq):
    _weight: MatMul
    _bias: Bias

    def __init__(self, weight=True, bias=True, dims=None):
        if dims is not None:
            assert isinstance(weight, bool) and weight
            assert isinstance(bias, bool)
            # assert isinstance(dims, List[int] | FromDims)
            assert isinstance(dims, List[int])
            weight = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(*dims)))
            bias = nn.Parameter(torch.zeros(dims[-1]))
        if isinstance(weight, nn.Parameter):
            weight = MatMul(weight)
        if isinstance(bias, nn.Parameter):
            bias = Bias(bias)
        assert isinstance(weight, MatMul)
        super().__init__(_weight=weight, _bias=bias)

    @property
    def weight(self):
        return self._weight.weight

    @property
    def bias(self):
        return self._bias.bias


class Linear(saeco.core.module.Module):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    def forward(self, x, *, cache: Cache, **kwargs):
        cache.x = x
        return x @ self.W + self.bias


@dataclass
class CacheLayerConfig:
    d_in: int = 0
    d_out: int = 0
    inst: Optional[List[int]] = None


class CacheLinear(saeco.core.module.Module):
    def __init__(
        self,
        W: Float[Tensor, "*inst d_in d_out"],
        bias: Union[Float[Tensor, "*#inst d_out"], bool],
        b_in=None,
        nonlinearity=torch.nn.ReLU(),
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = W.shape[-2]
        self.d_out = W.shape[-1]
        self.inst = W.shape[:-2]
        assert isinstance(W, nn.Parameter)
        assert isinstance(bias, nn.Parameter) or isinstance(bias, bool) or bias is None
        self.W = W
        self.ndim_inst = self.W.ndim - 2
        self.b = bias
        if b_in is None:
            self.b_pre = 0
        else:
            assert isinstance(b_in, nn.Parameter | bool)
            if isinstance(b_in, nn.Parameter):
                self.b_pre = b_in
            else:
                assert b_in
                b_in = nn.Parameter(torch.zeros(*self.inst, self.d_in))
        self.nonlinearity = nonlinearity
        # self.W = nn.Parameter(W)

    def forward(self, x, *, cache: Cache, **kwargs):
        cache.x = x
        x = x.view(
            *(x.shape[:1] + (1,) * (self.ndim_inst - (x.ndim - 2)) + x.shape[1:])
        )
        # TODO become fully satisfied w the squeezing
        mul = (x + self.b_pre).unsqueeze(-2) @ self.W
        cache.pre_acts = (pre_acts := mul.squeeze(-2) + self.b)
        cache.acts = (
            acts := (
                self.nonlinearity(pre_acts, cache=cache)
                if isinstance(self.nonlinearity, saeco.core.module.Module)
                else self.nonlinearity(pre_acts)
            )
        )
        return acts

    @classmethod
    def from_dims(
        cls, d_in, d_out, bias=True, b_in=False, inst=tuple(), cfg=None, **kwargs
    ):
        if cfg is None:
            cfg = CacheLayerConfig(d_in, d_out, inst)
        W = torch.nn.init.kaiming_uniform_(torch.empty(*inst, d_in, d_out))
        b_out = torch.zeros(*inst, d_out) if bias else 0
        b_in = torch.zeros(*inst, d_in) if b_in else None
        return cls(W, b_out, b_in, cfg=cfg, **kwargs)

    @classmethod
    def from_cfg(cls, cfg: CacheLayerConfig):
        return cls.from_dims(cfg.d_in, cfg.d_out, inst=cfg.inst, cfg=cfg)
