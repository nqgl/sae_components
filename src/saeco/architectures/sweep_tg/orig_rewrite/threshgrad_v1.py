from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, wraps
from typing import Protocol, Self

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp.autocast_mode import custom_bwd, custom_fwd
from torch.autograd.function import FunctionCtx

from saeco.components.penalties.l0targeter import L0Targeting
from saeco.sweeps import SweepableConfig


def shrinkgrad_adjustment(errors, leniency, dd, b):
    return errors * (leniency * 2 / dd / b)


def sig_grad(x):
    sig = x.sigmoid()
    return sig * (1 - sig)


windows = {"sig": sig_grad}


class WriteOnceBox[T]:
    def __init__(self):
        self.val = ...

    def put(self, val: T) -> None:
        assert self.val is ...
        assert val is not ...
        self.val = val

    def get(self) -> T:
        assert self.val is not ...
        return self.val


class GTCtxProto(Protocol):
    gate_post: WriteOnceBox[Tensor]
    noise: WriteOnceBox[Tensor]
    leniency: float
    d_data: int
    saved_tensors: tuple[Tensor, ...]
    grad_window: Callable[[Tensor], Tensor]

    def save_for_backward(self, *tensors: Tensor): ...


@dataclass
class GradCtx:
    mag: Tensor
    gate: Tensor
    gate_pre: Tensor
    noise: WriteOnceBox[Tensor]
    d_data: int
    leniency: float
    gate_post: WriteOnceBox[Tensor]
    grad_window: Callable[[Tensor], Tensor]

    @classmethod
    def unpack(cls, ctx: GTCtxProto) -> Self:
        gate_pre, mag, gate = ctx.saved_tensors

        return cls(
            mag=mag,
            gate=gate,
            gate_pre=gate_pre,
            noise=ctx.noise,
            d_data=ctx.d_data,
            leniency=ctx.leniency,
            gate_post=ctx.gate_post,
            grad_window=ctx.grad_window,
        )

    def pack_ctx_(
        self,
        ctx: GTCtxProto,
    ):
        ctx.save_for_backward(
            self.gate_pre,
            self.mag,
            self.gate,
        )
        ctx.noise = self.noise
        ctx.gate_post = self.gate_post
        ctx.leniency = self.leniency
        ctx.d_data = self.d_data
        ctx.grad_window = self.grad_window

    def calculate_adjustment(self, deviation: Tensor, leniency: float | None = None):
        return shrinkgrad_adjustment(
            deviation,
            leniency=self.leniency if leniency is None else leniency,
            dd=self.d_data,
            b=self.gate_pre.shape[0],
        )


class ThreshGateConfigThing(SweepableConfig):
    modify_grad_output: bool = False
    offgrad_mask_only_on_positive_noise: bool = True
    eps_0: float = 0.0
    fixed_off_leniency: float | None = None
    out1_has_off_adjustment: bool = False
    out1_has_on_adjustment: bool = False

    def _modify_grad_output(self, grad_output: Tensor, *, gcx: GradCtx):
        if self.modify_grad_output:
            return torch.where(gcx.mag != 0, grad_output / (gcx.mag + self.eps_0), 0)
        return grad_output

    def _make_offgrad_mask(self, grad_output: Tensor, gcx: GradCtx):
        mask = (~gcx.gate) & (grad_output != 0)
        if self.offgrad_mask_only_on_positive_noise:
            return mask & (gcx.noise.get() > 0)
        return mask

    def backward(self, gcx: GradCtx, grad_output: Tensor):
        grad_output = self._modify_grad_output(grad_output, gcx=gcx)

        grad_gate = (gcx.noise.get() < 0) & (gcx.gate_post.get() > 0)
        offgrad_mask = self._make_offgrad_mask(grad_output, gcx=gcx)
        adjustment = gcx.calculate_adjustment(-gcx.noise.get())
        off_adjustment = gcx.calculate_adjustment(
            gcx.mag, leniency=self.fixed_off_leniency
        )

        out0 = torch.where(grad_gate, grad_output + adjustment, 0)
        out1 = self._make_out1(
            grad_output=grad_output,
            offgrad_mask=offgrad_mask,
            adjustment=adjustment,
            off_adjustment=off_adjustment,
        )
        out = out0 + out1
        return (
            out * gcx.grad_window(gcx.gate_pre),
            None,
            None,
            None,
            None,
        )

    def _make_out1(
        self,
        *,
        grad_output: Tensor,
        offgrad_mask: Tensor,
        adjustment: Tensor,
        off_adjustment: Tensor,
    ):
        assert self.out1_has_off_adjustment or self.out1_has_on_adjustment
        out_grad = grad_output
        if self.out1_has_off_adjustment:
            out_grad += off_adjustment
        if self.out1_has_on_adjustment:
            out_grad += adjustment
        return torch.where(offgrad_mask, out_grad, 0)


def make_threshgate_autograd_function(tg_config: ThreshGateConfigThing, d_data: int):
    class ThreshgateAutogradFunction(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda")
        def forward(
            ctx: GTCtxProto,
            gate_pre: Tensor,
            gate_post: WriteOnceBox[Tensor],
            noise: WriteOnceBox[Tensor],
            mag: Tensor,
            leniency: float,
            grad_window: Callable[[Tensor], Tensor] = sig_grad,
        ):
            gate = gate_pre > 0
            GradCtx(
                mag=mag,
                gate=gate,
                gate_pre=gate_pre,
                d_data=d_data,
                leniency=leniency,
                noise=noise,
                gate_post=gate_post,
                grad_window=grad_window,
            ).pack_ctx_(ctx)

            return gate.float()

        @staticmethod
        @custom_bwd(device_type="cuda")
        def backward(ctx: GTCtxProto, *grad_outputs: Tensor):
            (grad_output,) = grad_outputs
            gcx = GradCtx.unpack(ctx)
            return tg_config.backward(gcx, grad_output)

    def apply(
        gate_pre: Tensor,
        gate_post: WriteOnceBox[Tensor],
        noise: WriteOnceBox[Tensor],
        mag: Tensor,
        leniency: float,
        grad_window: Callable[[Tensor], Tensor] = sig_grad,
    ) -> Tensor:
        t = ThreshgateAutogradFunction.apply(
            gate_pre, gate_post, noise, mag, leniency, grad_window
        )
        assert isinstance(t, Tensor)
        return t

    return apply


@dataclass
class GateFunctionConfig:
    training: bool
    uniform_noise: bool
    noise_mult: float
    leniency: float
    exp_mag: bool
    mag_scale_noise: bool = False  # TODO calculate grad right if doing this
    gate_noise: bool = False
    scale_penalty: bool = True
    dd: int = 768  # TODO
    p_off_noise: float = 0.2
    window_fn: Callable[[Tensor], Tensor] = sig_grad  # TODO to str

    def make_noise(self, mag: Tensor):
        return (
            torch.where(
                (mag > 0),
                (0.5 - torch.rand_like(mag))
                if self.uniform_noise
                else torch.randn_like(mag),
                0,
            )
            * self.noise_mult
            if self.training
            else torch.zeros_like(mag)
        ).cuda()


@dataclass
class GateFunc:
    tb_backward_config_thing: ThreshGateConfigThing
    gate_cfg: GateFunctionConfig

    @cached_property
    def autograd_gate_fn(self):
        return make_threshgate_autograd_function(
            self.tb_backward_config_thing, d_data=self.gate_cfg.dd
        )

    def gate(
        self,
        mag: Tensor,
        gate_pres: list[Tensor],
        penalty_fn=None,
    ):
        mag = mag.relu() if not self.gate_cfg.exp_mag else mag.exp()
        # box these two since we don't know them until we get the output of gates
        future_gate_post: WriteOnceBox[Tensor] = WriteOnceBox()
        noise_box: WriteOnceBox[Tensor] = WriteOnceBox()

        gates: list[Tensor] = [
            self.autograd_gate_fn(
                gate_pre=gate_pre,
                gate_post=future_gate_post,
                noise=noise_box,
                mag=mag,
                leniency=self.gate_cfg.leniency,
            )
            for gate_pre in gate_pres
        ]

        gate = gates[0]
        for g in gates[1:]:
            gate = gate * g
        future_gate_post.put(gate)
        assert gate.dtype is torch.bool

        noise = self.gate_cfg.make_noise(mag)
        off_noise_mask = torch.rand_like(mag) < self.gate_cfg.p_off_noise

        if self.gate_cfg.mag_scale_noise:
            noise *= mag
        noise = torch.where(gate | off_noise_mask, noise, 0)
        noise_box.put(noise)

        if penalty_fn is not None:
            if self.gate_cfg.scale_penalty:
                penalty_fn(gate * mag.detach())
            else:
                penalty_fn(gate * (mag > 0))
                # condition on mag>0?
                # yeah
        if self.gate_cfg.gate_noise:
            out = gate * (mag + noise)
        else:
            out = gate * mag + noise
        return out


def gate(
    mag: Tensor,
    gate_pres: list[Tensor],
    cfg: GateFunctionConfig,
    GT_fn,
    penalty_fn=None,
):
    future_gate_post = torch.ones_like(mag)
    mag = mag.relu() if not cfg.exp_mag else mag.exp()
    gate = None
    noise_mask = torch.rand_like(mag) < cfg.p_off_noise

    noise = cfg.make_noise(mag)
    for gate_pre in gate_pres:
        g = GT_fn(gate_pre, future_gate_post, noise, mag, cfg.leniency)
        if gate is None:
            gate = g
        else:
            gate = gate * g
    assert gate is not None
    with torch.no_grad():
        future_gate_post[:] = gate[:]
        noise[:] = torch.where((gate > 0) | noise_mask, noise, 0)
    if penalty_fn is not None:
        penalty_fn(gate * mag.detach())
    out = gate * mag + noise
    return out


import saeco.core as cl
from saeco.components.features import FeaturesParam
from saeco.initializer import Initializer


class Config(SweepableConfig):
    pre_bias: bool = False
    uniform_noise: bool = True
    noise_mult: float = 0.1
    exp_mag: bool = True
    mag_weights: bool = False
    window_fn: str = "sig"
    decay_l1: bool = True
    leniency_targeting: bool = False
    leniency: float = 1


class BinaryEncoder(cl.Module):
    def __init__(
        self,
        cfg: Config,
        init: Initializer,
        penalty=None,
        apply_targeting_externally=False,
        signed_mag: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.mag = init._encoder.new_bias()
        self.mag.data -= 1
        # self.mag.data += 2
        self.gate = init.encoder
        self.targeting = L0Targeting(
            init.l0_target,
            scale=cfg.leniency,
            increment=0.0002 if cfg.leniency_targeting else 0.0,
        )
        self.GT_fn = ThreshgateAutogradFunction
        self.apply_targeting_externally = apply_targeting_externally
        self.signed_mag = signed_mag
        self.penalty = penalty

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        cache.leniency = ...
        cache.leniency = self.targeting.value
        if (
            (not self.cfg.leniency_targeting)
            and cache._ancestor.has.trainstep
            and cache._ancestor.trainstep <= 5000
        ):
            self.targeting.value = cache._ancestor.trainstep / 5000
        mag = self.mag.unsqueeze(0).expand(x.shape[0], -1)
        cfg = GateFunctionConfig(
            exp_mag=not self.signed_mag,
            training=self.training,
            uniform_noise=self.cfg.uniform_noise,
            noise_mult=self.cfg.noise_mult,
            leniency=(
                self.targeting.value
                * (
                    1
                    if not cache._ancestor.has.trainer
                    else cache._ancestor.trainer.cfg.schedule.lr_scale(
                        cache._ancestor.trainstep
                    )
                )
            ),
        )

        out = gate(
            mag.abs() if self.signed_mag else mag,
            [self.gate(x)],
            GT_fn=self.GT_fn,
            cfg=cfg,
            penalty_fn=cache(self).penalty if self.penalty is not None else None,
        )
        if not self.apply_targeting_externally:
            cache(self).targeting(out)
        out = out * torch.sign(mag) if self.signed_mag else out
        return out

    @property
    def features(self):
        return {"mag": FeaturesParam(self.mag, 0, "bias")}


class GTTest(cl.Module):
    def __init__(self, cfg: Config, init: Initializer, targeting=True):
        super().__init__()
        self.cfg = cfg
        self.mag = nn.Linear(init.d_data, init.d_dict)
        # self.mag.data += 2
        self.gate = init.encoder
        self.gate.weight.grad
        self.targeting = L0Targeting(
            init.l0_target,
            scale=cfg.leniency,
            increment=0.0001 if cfg.leniency_targeting else 0.0,
        )
        self.GT_fn = ThreshgateAutogradFunction

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        cache.leniency = ...
        cache.leniency = self.targeting.value
        gate_fn_cfg = GateFunctionConfig(
            exp_mag=self.cfg.exp_mag,
            training=self.training,
            uniform_noise=self.cfg.uniform_noise,
            noise_mult=self.cfg.noise_mult,
            leniency=self.targeting.value,
        )

        out = gate(
            self.mag(x),
            gate_pres=[self.gate(x)],
            GT_fn=self.GT_fn,
            cfg=gate_fn_cfg,
        )
        cache(self).targeting(out)
        return out

    @property
    def features(self):
        return {
            "mag": FeaturesParam(self.mag.weight, 0, "other"),
            "mag_bias": FeaturesParam(self.mag.bias, 0, "bias"),
        }


class GTMulti(cl.Module):
    def __init__(
        self, cfg: Config, init: Initializer, d_in_override=None, targeting=True
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.mag_weights:
            self.mag = init._encoder.new_bias()
            self.mag.data -= 1
        # self.mag.data += 2
        self.gate = init.encoder
        self.targeting = L0Targeting(
            init.l0_target,
            scale=cfg.leniency,
            increment=0.0001 if cfg.leniency_targeting else 0.0,
        )
        self.GT_fn = ThreshgateAutogradFunction

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        cache.leniency = ...
        cache.leniency = self.targeting.value
        if (
            (not self.cfg.leniency_targeting)
            and cache._ancestor.has.trainstep
            and cache._ancestor.trainstep <= 5000
        ):
            self.targeting.value = cache._ancestor.trainstep / 5000
        cfg = GateFunctionConfig(
            exp_mag=True,
            training=self.training,
            uniform_noise=self.cfg.uniform_noise,
            noise_mult=self.cfg.noise_mult,
            leniency=self.targeting.value,
        )
        out = gate(
            self.mag.unsqueeze(0).expand(x.shape[0], -1),
            [self.gate(x)],
            GT_fn=self.GT_fn,
            cfg=cfg,
        )
        cache(self).targeting(out)
        return out

    @property
    def features(self):
        return {"mag": FeaturesParam(self.mag, 0, "bias")}
