import saeco.core as cl
import saeco.components as co
import einops
import torch


class Gated(cl.Parallel):
    def __init__(self, *, gate, mag=None, cond=lambda x: x > 0):
        super().__init__(
            gate=gate,
            mag=mag or cl.ops.Identity(),
            _support_parameters=True,
            _support_modules=True,
        )

        self.cond = cond
        self.reduce(lambda g, v: self.cond(g) * v)

    def aux(self):
        return self.gate

    def full(self):
        return self


def hl2ll(hl, bf):
    return einops.repeat(hl, "q i -> q (i bf)", bf=bf)


def struc(ll, bf):
    return einops.rearrange(ll, "q (i bf) -> q i bf", bf=bf)


def unstruc(ll):
    return einops.rearrange(ll, "q i bf -> q (i bf)")


def l1normalize(ll, bf):
    structured = struc(ll, bf)
    return unstruc(structured / structured.sum(dim=-1, keepdim=True))


class SafeNormalized(cl.Module):
    def __init__(self, bf, norm, noise=None, detach=True):
        super().__init__()
        self.bf = bf
        self.norm = norm
        self.noise = noise or (lambda x: x)
        self.detach = detach

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        if self.detach:
            x = x.detach()
        structured = struc(self.noise(x), self.bf)
        norm = self.norm(structured)
        out = unstruc(structured / norm)
        return torch.where(~out.isnan(), out, 0)
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
    def __init__(self, gate: cl.Module, branching_factor):
        super().__init__()
        self.gate = gate
        self.branching_factor = branching_factor

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return hl2ll(cache(self).gate(x), self.branching_factor)


class HGated:
    def __init__(self, hl, ll, bf, normalization=2):
        self.hl = cl.ReuseForward(
            HGateExpand(
                gate=cl.Seq(
                    gate=hl,
                    metrics=co.metrics.Metrics(L0=co.metrics.L0(), L1=co.metrics.L1()),
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

    def aux(self):
        return cl.ops.MulParallel(
            gate_aux=self.hl,
            directions=cl.Seq(
                ll=self.ll,
                normalization=self.normalization,
            ),
        )

    def full(self):
        return Gated(
            gate=self.hl,
            mag=self.ll,
        )
