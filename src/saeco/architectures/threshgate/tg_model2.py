import torch

import torch.nn as nn

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


class HSMix(cl.Module):
    def __init__(self, p_soft: float, mag_noise=0.1):
        super().__init__()
        self.p_soft = p_soft
        self.hard = ReuseForward(co.ops.Lambda(lambda x: (x > 0).half()))

        if mag_noise is None:
            self.soft = nn.Sigmoid()
        else:
            self.soft = Seq(
                co.ops.Lambda(lambda x: x + (torch.rand_like(x) - 0.5) * mag_noise),
                nn.Sigmoid(),
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


class Config(SweepableConfig):
    pre_bias: bool = False
    # detach: bool = Swept(True, False)
    bf: int = 32
    num_layers: int = 3
    l0_target_ratio: float = 2
    use_enc_bias: int = True
    use_relu: bool = True
    noise_mag: float = 0.1
    p_soft: float = 0.1
    end_l1_penalty: float = 0.01
    # decay_l1_penalty: float =


class MultiGate(cl.Module):
    def __init__(
        self,
        n_sections=3,
        targets: float | list[float] = 5,
        use_relu=True,
        noise_mag=0.1,
        p_soft=0.1,
        end_l1_penalty=0.01,
    ):
        super().__init__()
        if isinstance(targets, float):
            targets = [targets] * (n_sections - 1)
        assert len(targets) == n_sections - 1
        self.n_sections = n_sections
        self.all_acts = ReuseForward(
            cl.Router(
                *(
                    [nn.ReLU() if use_relu else nn.Identity()]
                    + [
                        ReuseForward(HSMix(p_soft=p_soft, mag_noise=noise_mag))
                        for _ in range(n_sections - 1)
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
            begin=0.3,
            end=15_000,
            begin_scale=0,
            end_scale=end_l1_penalty,
        )

    def forward(self, pre_acts_list: list[torch.Tensor], *, cache: cl.Cache, **kwargs):
        assert len(pre_acts_list) == self.n_sections

        cache(self).update_penalty_l0s(pre_acts_list)
        cache(self).apply_penalty(pre_acts_list)
        out = torch.where(
            cache(self).noised_mask(pre_acts_list),
            cache(self).out_det(pre_acts_list),
            cache(self).out(pre_acts_list),
        )
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
        cfg.num_layers,
        [None, 100],
        use_relu=cfg.use_relu,
        noise_mag=cfg.noise_mag,
        p_soft=cfg.p_soft,
        end_l1_penalty=cfg.end_l1_penalty,
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
            lambda g, a: [
                torch.where(g, a, 0),
                torch.where(~g, a, 0),
            ]
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
