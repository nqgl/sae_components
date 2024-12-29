import torch

import torch.nn as nn
from typing import Callable

import saeco.components as co
import saeco.components.features.features as ft
from saeco.components.features.optim_reset import FeatureParamType
import saeco.core as cl
from saeco.core import ReuseForward
from saeco.initializer import Initializer, Tied
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)

from saeco.components.hooks.clipgrad import ClipGrad
from saeco.core import Seq

from saeco.misc import useif
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.components.penalties.l1_penalizer import (
    L0TargetingL1Penalty,
    LinearDecayL1Penalty,
)
from saeco.sweeps import SweepableConfig, Swept
import einops
from saeco.components.hierarchical import hl2ll
from torch.cuda.amp import custom_bwd, custom_fwd
from saeco.components.jumprelu.kernels_fns import rect, gauss


def thresh_shrink(shrink_amount, eps, mult_by_shrank):
    class ThreshShrinkSoft(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, x):
            shrink = torch.randn_like(x) * shrink_amount
            gate = x > 0
            ctx.save_for_backward(shrink, x)
            return torch.where(gate, 1 - shrink, 0)

        @staticmethod
        @custom_bwd
        def backward(ctx: torch.Any, grad_output):
            (shrank, x) = ctx.saved_tensors
            if mult_by_shrank:
                return 1 / eps * gauss((x) / eps) * grad_output * shrank
            return 1 / eps * gauss((x) / eps) * grad_output

            return shrank * grad_output

    return ThreshShrinkSoft.apply


class HardSoftConfig(SweepableConfig):
    noise_mag: float | None = 0.1
    p_soft: float = 0.1
    eps: float = 0.03
    mult_by_shrank: bool = False
    post_noise: float | None = 0.1
    uniform_noise: bool = True
    noise_scale: float = 0.1
    gate_backwards: bool = True


class HSMix(cl.Module):
    def __init__(self, cfg: HardSoftConfig, soft_cls=nn.Sigmoid):
        super().__init__()
        self.p_soft = cfg.p_soft
        self.hard = ReuseForward(co.ops.Lambda(lambda x: (x > 0).half()))

        if cfg.noise_mag is None:
            self.soft = soft_cls()
        else:
            self.soft = Seq(
                co.ops.Lambda(lambda x: x + (torch.rand_like(x) - 0.5) * cfg.noise_mag),
                soft_cls(),
            )
        self.rand = ReuseForward(
            co.ops.Lambda(lambda x: torch.rand_like(x) < self.p_soft)
        )

    def forward(self, acts, *, cache: cl.Cache, **kwargs):
        if not self.training:
            return cache(self).hard(acts)
        return torch.where(
            cache(self).rand(acts), cache(self).soft(acts), cache(self).hard(acts)
        )


class ThreshHSMix(HSMix):
    def __init__(self, cfg: HardSoftConfig):
        super().__init__(
            cfg,
            soft_cls=lambda: co.ops.Lambda(
                thresh_shrink(0.1, eps=cfg.eps, mult_by_shrank=cfg.mult_by_shrank)
            ),
        )


def hs_from_soft(soft_cls):
    class CustHS(HSMix):
        def __init__(self, cfg: HardSoftConfig):
            super().__init__(
                cfg,
                soft_cls=soft_cls,
            )

    return CustHS


class RandSig_fn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x):
        sig = x.sigmoid()
        gated = torch.rand_like(x) < sig
        ctx.save_for_backward(x)
        return torch.where(gated, 0.9, 0).to(x.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * x.sigmoid() * (1 - x.sigmoid())


class RandSig(nn.Module):
    def forward(self, x):
        return RandSig_fn.apply(x)


RandSigHS = hs_from_soft(RandSig)


def fromnoise_op(
    noise_op,
    uniform_noise=False,
    noise_scale=0.1,
    gate_backwards=True,
):
    class NormalrandSig_fn(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, x):
            sig = x.sigmoid()
            gated = torch.rand_like(x) < sig
            noise = (
                (torch.rand_like(x) - 0.5) * (12**0.5)  # var-> 1
                if uniform_noise
                else torch.randn_like(x)
            )
            noise = noise * noise_scale
            ctx.save_for_backward(sig, noise, gated)

            return torch.where(gated, 1 - noise, 0).to(x.dtype)

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            (sig, noise, gated) = ctx.saved_tensors
            grad = noise_op(noise) * grad_output * sig * (1 - sig)
            if gate_backwards:
                return torch.where(gated, grad, 0)
            return grad

    class NormalrandSig(nn.Module):
        def forward(self, x):
            return NormalrandSig_fn.apply(x)

    return NormalrandSig


class NormalrandSigHS(HSMix):
    def __init__(self, cfg: HardSoftConfig):
        super().__init__(
            cfg,
            soft_cls=fromnoise_op(
                torch.relu,
                uniform_noise=cfg.uniform_noise,
                noise_scale=cfg.noise_scale,
                gate_backwards=cfg.gate_backwards,
            ),
        )


NormalrandSigHS
ExpNormalrandSigHS = hs_from_soft(fromnoise_op(torch.exp))
SigNormalrandSigHS = hs_from_soft(fromnoise_op(torch.sigmoid))
IgnoreNoiseGradSigHS = hs_from_soft(fromnoise_op(lambda x: 1))

hs_types = {
    "sigmoid": HSMix,
    "shrankthresh": ThreshHSMix,
    "randsig": RandSigHS,
    "normrandsig": NormalrandSigHS,
    "expnormrandsig": ExpNormalrandSigHS,
    "signormrandsig": SigNormalrandSigHS,
    "ignoreshrank": IgnoreNoiseGradSigHS,
}


class Config(SweepableConfig):
    hs_cfg: HardSoftConfig = HardSoftConfig()
    pre_bias: bool = False
    bf: int = 32
    detach: bool = True
    num_layers: int = 3
    # l0_target_ratio: float = 2
    use_enc_bias: int = True
    use_relu: bool = True
    end_l1_penalty: float = 0.01
    hs_type: str = ""


def expand_to_list(i, n):
    if isinstance(i, list):
        assert len(i) == n
        return i
    return [i] * n


class MultiGate(cl.Module):
    def __init__(
        self,
        hs_cfg: HardSoftConfig | list[HardSoftConfig],
        targets: float | list[float],
        hs_generator: Callable | list[Callable] = HSMix,
        n_sections=3,
        use_relu=True,
        end_l1_penalty=0.01,
        detach=True,
    ):
        super().__init__()
        targets = expand_to_list(targets, n_sections - 1)
        hs_generator = expand_to_list(hs_generator, n_sections - 1)
        hs_cfg = expand_to_list(hs_cfg, n_sections - 1)
        self.detach = detach
        self.n_sections = n_sections
        self.all_acts = ReuseForward(
            cl.Router(
                *(
                    [nn.ReLU() if use_relu else nn.Identity()]
                    + [
                        ReuseForward(hs_generator[i - 1](cfg=hs_cfg[i - 1]))
                        for i in range(1, n_sections)
                    ]
                )
            ).reduce(lambda *l: l)
        )
        self.acts_0 = ReuseForward(
            cl.Seq(
                self.all_acts,
                co.ops.Lambda(lambda l: l[0]),
            )
        )

        self.gates_only = ReuseForward(
            cl.Seq(
                self.all_acts,
                co.ops.Lambda(lambda l: l[1:]),
                cl.Router(*[cl.ops.Identity() for _ in range(n_sections - 1)]).reduce(
                    lambda a, b: a * b, binary=True
                ),
            ),
        )

        self.out = ReuseForward(
            cl.Parallel(
                self.acts_0,
                self.gates_only,
            ).reduce(lambda acts, gates: acts * gates)
        )

        self.out_det = ReuseForward(
            cl.Parallel(
                co.ops.Detached(self.acts_0),
                self.gates_only,
            ).reduce(lambda acts, gates: acts * gates)
        )

        self.penalizers = [None] + [
            (
                L0TargetingL1Penalty(
                    target=t or 0, scale=2, increment=0 if t is None else 0.0003
                )
            )
            for i, t in enumerate(targets)
        ]

        def slice_after(i):
            def _slice(l):
                return l[i:]

            return _slice

        # def bound_index(l, i):
        #     def index

        self.update_penalty_l0s = Seq(
            calculate_hard=cl.Parallel(
                *(
                    [self.acts_0]
                    + [
                        cl.Seq(
                            mask=cl.Seq(
                                co.ops.Lambda(slice_after(i)),
                                cl.Router(
                                    *[
                                        self.all_acts.module[j].module.hard
                                        for j in range(i, n_sections)
                                    ]
                                ).reduce(
                                    lambda a, b: a * b,
                                    binary=True,
                                    binary_initial_value=1,
                                ),
                            ),
                            update_penalty=co.metrics.ActMetrics(
                                f"acts_{i}",
                                update=co.ops.Lambda(self.penalizers[i].update_l0),
                            ),
                        )
                        for i in range(1, n_sections)
                    ]
                )
            ).reduce(lambda a, b, *z: a * b),
            freqs=EMAFreqTracker(),
            metrics=co.metrics.ActMetrics(),
            # penalty=L1Penalty(scale=1),
        )

        self.apply_penalty = Seq(
            cl.Parallel(
                out_det=Seq(
                    self.out,
                    co.ops.Lambda(
                        lambda x: x.detach(),
                    ),
                ),
                gates=self.all_acts,
            ).reduce(
                lambda out_det, gates: [
                    out_det * (1 + gates[i] - gates[i].detach().half())
                    for i in range(1, n_sections)
                ],
            ),
            cl.Router(*[self.penalizers[i] for i in range(1, n_sections)]).reduce(
                lambda *l: None
            ),
        )

        self.noised_mask = Seq(
            co.ops.Lambda(lambda l: l[1:]),
            cl.Router(
                *[self.all_acts.module[i].module.rand for i in range(1, n_sections)]
            ).reduce(
                lambda a, b: a | b if a is not None else b,
                binary=True,
                binary_initial_value=None,
            ),
        )
        self.penalty = LinearDecayL1Penalty(
            begin_scale=0.3,
            end=25_000,
            begin=0,
            end_scale=end_l1_penalty,
        )

    def forward(self, pre_acts_list: list[torch.Tensor], *, cache: cl.Cache, **kwargs):
        assert len(pre_acts_list) == self.n_sections

        cache(self).update_penalty_l0s(pre_acts_list)
        cache(self).apply_penalty(pre_acts_list)
        if self.detach:
            out = torch.where(
                cache(self).noised_mask(pre_acts_list),
                cache(self).out_det(pre_acts_list),
                cache(self).out(pre_acts_list),
            )
        else:
            out = cache(self).out(pre_acts_list)
        cache(self).penalty(out)
        return out


class MGFeaturesParam(ft.FeaturesParam):
    def __init__(
        self,
        param: nn.Parameter,
        feature_index,
        fptype: FeatureParamType | None = None,
        resampled=True,
        reset_optim_on_resample=True,
    ):
        super().__init__(
            param, feature_index, fptype, resampled, reset_optim_on_resample
        )


def zero_normal_enc_bias_only_fn_factory(encoder: co.LinEncoder, d_dict):
    @co.ops.Lambda
    @torch.no_grad()
    def zero_normal_enc_bias_only(x):
        encoder.features["bias"].features[:d_dict].zero_()
        return x

    return zero_normal_enc_bias_only


def multigate_sae(
    init: Initializer,
    cfg: Config,
):

    init._encoder.d_out *= cfg.num_layers
    init._decoder._weight_tie = None
    init.decoder
    decx3 = init.decoder.weight.data.repeat(1, cfg.num_layers).transpose(-2, -1)
    init._encoder.bias = cfg.use_enc_bias > 0

    init._encoder._weight_tie = Tied(
        decx3 + torch.randn_like(decx3) * 0.03,
        Tied.TO_VALUE,
        "weight",
    )
    # encoder = init.encoder
    # encoder.stored_fake_res_weight = nn.Parameter(
    #     init.encoder.wrapped.weight[: init.d_dict]
    # )
    # encoder.get_weight = lambda: encoder.stored_fake_res_weight
    # encoder.stored_fake_res_bias = nn.Parameter(
    #     init.encoder.wrapped.bias[: init.d_dict]
    # )
    # encoder.get_bias = lambda: encoder.stored_fake_res_bias
    multigate = MultiGate(
        n_sections=cfg.num_layers,
        targets=[None, 100],
        use_relu=cfg.use_relu,
        hs_generator=hs_types[cfg.hs_type],
        hs_cfg=cfg.hs_cfg,
        end_l1_penalty=cfg.end_l1_penalty,
        detach=cfg.detach,
    )
    model = Seq(
        **useif(
            cfg.use_enc_bias == 1,
            update_enc_bias=zero_normal_enc_bias_only_fn_factory(
                init.encoder, init.d_dict
            ),
        ),
        encoder=Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            lin=init.encoder,
            split=co.ops.Lambda(lambda x: torch.split(x, init.d_dict, dim=-1)),
        ),
        gate_x=cl.Parallel(
            noise_gate=multigate.noised_mask,
            acts=multigate,
        ).reduce(
            (
                lambda g, a: [
                    torch.where(g, a, 0),
                    torch.where(~g, a, 0),
                ]
            )
            if cfg.detach
            else (lambda g, a: [torch.zeros_like(a), a])
        ),
        dec_router=cl.Router(
            det_dec=init._decoder.detached_no_bias,
            decoder=ft.OrthogonalizeFeatureGrads(
                ft.NormFeatures(
                    init.decoder,
                ),
            ),
        ).reduce(lambda a, b: a + b),
    )

    # losses
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model, num_expeted=cfg.num_layers),
    )
    return [model], losses


from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=multigate_sae)
    tr.trainer.train()
    tr.trainer.save()


if __name__ == "__main__":
    do_sweep(True)
else:
    from .tg2_config import cfg, PROJECT
