import torch

import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.core import ReuseForward
from saeco.architectures.initialization.initializer import Initializer, Tied
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
                co.ops.Lambda(lambda x: x + torch.randn_like(x) * mag_noise),
                nn.Sigmoid(),
            )

    def forward(self, acts, *, cache: cl.Cache, **kwargs):
        if not self.training:
            return cache(self).hard(acts)
        rand = torch.rand_like(acts) < self.p_soft
        return torch.where(rand, cache(self).soft(acts), cache(self).hard(acts))


class MultiGate(cl.Module):
    def __init__(self, n_sections=3, targets: float | list[float] = 5):
        super().__init__()
        # self.gates = nn.ModuleList()
        eps = 1e-6
        if isinstance(targets, float):
            targets = [targets] * (n_sections - 1)
        assert len(targets) == n_sections - 1
        self.n_sections = n_sections
        self.gates = ReuseForward(
            cl.Router(
                *(
                    [nn.ReLU()]
                    + [ReuseForward(HSMix(0.1)) for _ in range(n_sections - 1)]
                )
            ).reduce(lambda *l: l)
        )

        self.out = ReuseForward(
            cl.Seq(
                self.gates,
                cl.Router(*[cl.ops.Identity() for _ in range(n_sections)]).reduce(
                    lambda a, b: a * b, binary=True
                ),
            )
        )
        self.penalizers = [None] + [
            L0TargetingL1Penalty(target=t, scale=256 ** (-i))
            for i, t in enumerate(targets)
        ]
        self.update_penalty_l0s = Seq(
            calculate_hard=cl.Parallel(
                *(
                    [co.ops.Lambda(lambda l: l[0].relu())]
                    + [
                        cl.Seq(
                            mask=cl.Seq(
                                co.ops.Lambda(lambda l: l[i - 1 :]),
                                cl.Router(
                                    *[
                                        self.gates.module[j].module.hard
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
                                update=co.ops.Lambda(
                                    lambda x: self.penalizers[i].update_l0(x)
                                ),
                            ),
                        )
                        for i in range(1, n_sections)
                    ]
                )
            ).reduce(lambda a, b, *z: a * b),
            freqs=EMAFreqTracker(),
            metrics=co.metrics.ActMetrics(),
            penalty=LinearDecayL1Penalty(begin_scale=0, end=5_000),
        )

        self.apply_penalty = Seq(
            cl.Parallel(
                out_det=Seq(
                    self.out,
                    co.ops.Lambda(
                        lambda x: x.detach(),
                    ),
                ),
                gates=self.gates,
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

    def forward(self, pre_acts_list: list[torch.Tensor], *, cache: cl.Cache, **kwargs):
        assert len(pre_acts_list) == self.n_sections
        out = cache(self).out(pre_acts_list)
        l0s = cache(self).update_penalty_l0s(pre_acts_list)
        cache(self).apply_penalty(pre_acts_list)
        return out


class Config(SweepableConfig):
    pre_bias: bool = Swept(True, False)
    detach: bool = Swept(True, False)
    bf: int = 32
    num_layers: int = 3
    l0_target_ratio: float = 2

    # decay_l1_penalty: float =


def multigate_sae(
    init: Initializer,
    cfg: Config,
):

    init._encoder.d_out *= cfg.num_layers
    init._decoder._weight_tie = None
    init.decoder
    decx3 = init.decoder.weight.data.repeat(1, cfg.num_layers).transpose(-2, -1)

    init._encoder._weight_tie = Tied(
        decx3 + torch.randn_like(decx3) * 0.03,
        Tied.TO_VALUE,
        "weight",
    )
    model = Seq(
        encoder=Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            lin=init.encoder,
            split=co.ops.Lambda(lambda x: torch.split(x, init.d_dict, dim=-1)),
            muligate=MultiGate(cfg.num_layers, [25, 80]),
        ),
        decoder=ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                init.decoder,
            ),
        ),
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


if __name__ == "__main__":
    do_sweep(True, "rand")
else:
    from .tg2_config import cfg, PROJECT
