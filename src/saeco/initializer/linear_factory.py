from saeco.core.basic_ops import Sub
import torch
from typing import Optional
import torch.nn as nn
from functools import cached_property
from saeco.misc import lazycall


class DetachedLinear(nn.Module):
    def __init__(self, lin, use_bias=True):
        super().__init__()
        self.lin = lin
        self.use_bias = use_bias

    def forward(self, x):
        return torch.nn.functional.linear(
            x,
            self.lin.weight.detach(),
            (
                self.lin.bias.detach()
                if (self.lin.bias is not None) and self.use_bias
                else None
            ),
        )


class InitSite:
    def __init__(self, site):
        self.site = site

    def __getattribute__(self, name: str) -> "InitSite":
        pass

    def __setattr__(self, name: str, value) -> None:
        pass


class BiasDetachedLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        return torch.nn.functional.linear(
            x,
            self.lin.weight,
            self.lin.bias.detach(),
        )


class Tied:
    INIT = 0
    TIED = 1
    TO_VALUE = 2
    INIT_FN = 3

    def __init__(self, target: "LinearFactory", tie_type, site: str):
        self.target = target
        self.tie_type = tie_type
        self.site = site

    def __call__(self, other: nn.Linear):
        if self.tie_type == self.TO_VALUE:
            dst_param = getattr(other, self.site)
            assert isinstance(self.target, torch.Tensor)
            dst_param.data[:] = self.target
            return
        if self.tie_type == self.INIT_FN:
            dst_param = getattr(other, self.site)
            o = self.target(dst_param.data)
            if o is not None:
                dst_param.data = o
            return
        src_param = getattr(self.target.raw, self.site)
        if self.tie_type == self.INIT:
            dst_param = getattr(other, self.site)
            assert (dst_param.data.shape == src_param.data.shape) or (
                dst_param.data.shape == src_param.data.transpose(-2, -1).shape
            )
            if (dst_param.data.shape == src_param.data.shape) and (
                dst_param.data.shape == src_param.data.transpose(-2, -1).shape
            ):
                print(
                    "WARNING: Tied initialization weights are the same shape. Ambiguous whether to transpose. Defaulting to transposing"
                )
                dst_param.data[:] = src_param.data.transpose(-2, -1)

            elif dst_param.data.shape == src_param.data.shape:
                dst_param.data[:] = src_param.data
            else:
                dst_param.data[:] = src_param.data.transpose(-2, -1)
        elif self.tie_type == self.TIED:
            dst_param = getattr(other, self.site)
            assert (dst_param.data.shape == src_param.data.shape) ^ (
                dst_param.data.shape == src_param.data.transpose(-2, -1).shape
            )
            if dst_param.data.shape == src_param.data.shape:
                dst_param.data = src_param.data
            else:
                dst_param.data = src_param.data.transpose(-2, -1)
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
        if self.bias and self._bias_tie is not None:
            self._bias_tie(lin)
        if self._linear_raw is not None:
            lin.weight.data[:] = self._linear_raw.weight.data
            if self.bias:
                lin.bias.data[:] = self._linear_raw.bias.data
        else:
            self._linear_raw = lin
        for w in self.wrappers:
            lin = w(lin)
        return lin

    def make_hierarchical(self, bf):
        assert self.d_out % bf == 0
        import einops

        lin = nn.Linear(self.d_in, self.d_out // bf, bias=True)
        ll = self.raw.weight.data
        # v = einops.rearrange(ll, "(i bf) q -> i bf q", bf=bf).sum(dim=-2)
        # v = einops.rearrange(ll, "(bf i) q -> bf i q", bf=bf).sum(dim=0)
        # lin.weight.data[:] = v * (ll.std() / v.std(dim=0, keepdim=True))
        # lin.bias.data[:] = lin.bias.data - 0.25
        return lin

    def get(self) -> nn.Linear:
        if self._linear is None:
            self._linear = self.make_new()
        return self._linear

    def new_bias(self) -> torch.Tensor:
        class temp:
            weight = self.lin.weight
            bias = torch.zeros(self.d_out)

        if self._bias_tie is not None:
            self._bias_tie(temp)
        return nn.Parameter(temp.bias)

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

    @property
    def detached_no_bias(self):
        return DetachedLinear(self.lin, use_bias=False)

    @property
    def biasdetached(self):
        return BiasDetachedLinear(self.lin)  # should this use raw?

    def tie_weights(self, other):
        assert self.unset
        # self.lin.weight = other.lin.weight
        assert self._weight_tie is None
        self._weight_tie = Tied(other, Tied.TIED, "weight")

    def tied_weights_init(self, other):
        assert self.unset
        assert self._weight_tie is None
        self._weight_tie = Tied(other, Tied.INIT, "weight")

    def const_init_bias(self, const=0):
        assert self.unset
        assert self._bias_tie is None
        self._bias_tie = Tied(torch.zeros(self.d_out) + const, Tied.TO_VALUE, "bias")
        # assert (self.lin.weight.data.shape == other.lin.weight.data.shape) ^ (
        #     self.lin.weight.data.shape == other.lin.weight.data.transpose(-2, -1).shape
        # )
        # if self.lin.weight.data.shape == other.lin.weight.data.shape:
        #     self.lin.weight.data[:] = other.lin.weight.data
        # else:
        #     self.lin.weight.data[:] = other.lin.weight.data.transpose(-2, -1)

    @lazycall
    def sub_bias(self) -> Sub:
        return Sub(self.lin.bias)
