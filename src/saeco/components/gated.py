import saeco.core as cl
import saeco.components as co
import einops
import torch
import torch.nn as nn
from saeco.initializer import Initializer
from saeco.misc import useif

SOFT_CLS = lambda x: cl.Seq(x, nn.Sigmoid())
SOFT_CLS = lambda module: cl.Seq(
    module, co.Lambda(lambda x: x + torch.randn_like(x) * 0.05), nn.Sigmoid()
)
SOFT_CLS = lambda module: cl.Seq(
    module,
    co.Lambda(
        lambda x: torch.where(
            torch.rand_like(x) > 0.2,
            torch.sigmoid(x) + torch.randn_like(x) * 0.05,
            (x > 0).float(),
        )
    ),
)
# SOFT_CLS = lambda module: cl.Seq(module, nn.Sigmoid())


class Gated(cl.Parallel):
    def __init__(self, *, gate, mag=None, thresh_cls=None):
        super().__init__(
            gate=cl.ReuseForward((thresh_cls or co.ops.Thresh)(gate)),
            mag=mag or cl.ops.Identity(),
            _support_parameters=True,
            _support_modules=True,
        )
        self.gate_aux = gate
        self.reduce(lambda g, v: g * v)

    def aux(self, i=0, full_mode=True, soft=False):
        assert i == 0
        if not soft:
            return self.gate_aux
        return cl.Seq(self.gate_aux, nn.ReLU())
        return Gated(
            gate=self.gate_aux,
            mag=co.Lambda(lambda x: x.detach(), self.mag),
            thresh_cls=SOFT_CLS,
        )

    def full(self, soft=False):
        if soft:
            self.full_soft()
        return self

    def full_soft(self, detach=True):
        # return self
        if detach:
            return Gated(
                gate=self.gate_aux,
                mag=co.Lambda(lambda x: x.detach(), self.mag),
                thresh_cls=SOFT_CLS,
            )
        return Gated(
            gate=self.gate_aux,
            mag=self.mag,
            thresh_cls=SOFT_CLS,
        )


class ClassicGated(Gated):
    def __init__(self, *, init: Initializer, penalize_inside_gate=False):
        enc_mag = cl.Seq(
            scaled_encoder=cl.ops.MulParallel(
                encoder=cl.ReuseForward(init.encoder),
                exp_r=co.ops.Lambda(
                    func=lambda x: torch.exp(x),
                    module=init.dict_bias(),
                ),
            ),
            bias=init.new_encoder_bias().resampled(),
            nonlinearity=nn.ReLU(),
        )

        enc_gate = cl.ReuseForward(
            cl.Seq(
                weight=cl.ReuseForward(init.encoder),
                bias=init.new_encoder_bias().resampled(),
                # nonlinearity=nn.ReLU(),
                base_gate_metrics=co.metrics.ActMetrics("Gate0"),
                **useif(
                    penalize_inside_gate,
                    penalty=co.L1Penalty(),
                ),
            )
        )

        super().__init__(gate=enc_gate, mag=enc_mag)


def hl2ll(hl, bf):
    return einops.repeat(hl, "b i -> b (i bf)", bf=bf)


def struc(ll, bf):
    return einops.rearrange(ll, "b (i bf) -> b i bf", bf=bf)


def unstruc(ll):
    return einops.rearrange(ll, "b i bf -> b (i bf)")


# def struc(ll, bf):
#     return einops.rearrange(ll, "b (bf i) -> b i bf", bf=bf)


# def unstruc(ll):
#     return einops.rearrange(ll, "b i bf -> b (bf i)")


class SafeNormalized(cl.Module):
    def __init__(self, bf, norm, noise=None, detach=True):
        super().__init__()
        self.bf = bf
        self.norm = norm
        self.noise = noise or (lambda x: x)
        self.detach = detach

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        # return x
        # if self.detach:
        #     x = x.detach()
        # return x
        structured = struc(self.noise(x), self.bf)
        norm = self.norm(structured)
        normed = structured / (norm + 1e-6)
        # normed = torch.where(norm > 1e-5, normed, 0)
        out = unstruc(normed)
        return torch.where(
            ~out.isnan(),
            out,
            # torch.sigmoid(out) * 0.01,
            (torch.relu(torch.rand_like(out) - 0.8)) * 0,
        )
        return out


def l1norm(structured):
    return structured.sum(dim=-1, keepdim=True)


def l2norm(structured):
    return structured.norm(dim=-1, keepdim=True)


class L1Normalize(SafeNormalized):
    def __init__(self, bf, noise=None):
        super().__init__(bf, l1norm, noise)


class L2Normalize(SafeNormalized):
    def __init__(self, bf, noise=None):
        super().__init__(bf, l2norm, noise)


class HGateExpand(cl.Module):
    def __init__(self, hgate: cl.Module, branching_factor):
        super().__init__()
        self.hgate = hgate
        self.branching_factor = branching_factor

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return hl2ll(cache(self).hgate(x), self.branching_factor)


class HGated:
    def __init__(self, hl, ll, bf, num, normalization=1):
        self.hl = cl.ReuseForward(
            HGateExpand(
                hgate=cl.Seq(
                    gate_enc=hl,
                    metrics=co.metrics.ActMetrics(f"HGate{num}"),
                ),
                branching_factor=bf,
            ),
        )
        self.ll = cl.ReuseForward(ll)
        self.bf = bf
        self.normalization = (
            {
                1: L1Normalize,
                2: L2Normalize,
            }[
                normalization
            ](bf)
            if isinstance(normalization, int)
            else normalization
        )

    def aux(self, i, full_mode=True, soft=False):
        if i == 0:
            return cl.ops.MulParallel(
                gate_aux=self.hl,
                directions=cl.Seq(
                    ll=self.ll.module.full(soft),
                    normalization=self.normalization,
                ),
            )
        if full_mode:
            return Gated(
                gate=self.hl,
                mag=self.ll.module.aux(i - 1, full_mode=full_mode, soft=soft),
                **useif(
                    soft,
                    thresh_cls=SOFT_CLS,
                ),
            )
        return self.ll.module.aux(i - 1, full_mode=full_mode, soft=soft)

    def full(self, soft=False):
        if soft:
            self.full_soft()
        return Gated(
            gate=self.hl,
            mag=self.ll.module.full(),
        )

    def full_soft(self):
        return Gated(
            gate=self.hl,
            mag=self.ll.module.full_soft(),
            thresh_cls=SOFT_CLS,
        )
