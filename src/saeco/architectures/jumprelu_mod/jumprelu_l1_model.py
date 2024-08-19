import torch

import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl

from saeco.initializer import Initializer
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
    SummedPenalties,
    LinearDecayL1Penalty,
)
from saeco.sweeps import SweepableConfig
from saeco.components.jumprelu.jumprelu_fn import HStep, JumpReLU, L0Penalty
from saeco.components.jumprelu.kernels_fns import kernels


class Config(SweepableConfig):
    eps: float
    kernel: str = "rect"
    pre_bias: bool = False
    thresh_initial_value: float = 0.0
    l1_begin_scale: float = 1.0
    l1_end: int = 20_000
    l1_end_scale: float = 0.0

    def get_kernel(self):
        return kernels[self.kernel]


def jumprelu_l1_sae(
    init: Initializer,
    cfg: Config,
):
    thresh = nn.Parameter(
        torch.zeros(init.d_dict) + cfg.thresh_initial_value
    )  # TODO resample thresh
    model = Seq(
        encoder=Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            lin=init.encoder,
            nonlinearity=JumpReLU(thresh=thresh, kernel=cfg.get_kernel(), eps=cfg.eps),
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        penalty=SummedPenalties(
            L0Penalty(thresh=thresh, kernel=cfg.get_kernel(), eps=cfg.eps),
            LinearDecayL1Penalty(
                end=cfg.l1_end,
                begin=5_000,
                begin_scale=cfg.l1_begin_scale,
                end_scale=cfg.l1_end_scale,
            ),
        ),
        decoder=ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                init.decoder,
            ),
        ),
    )

    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return models, losses


from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=jumprelu_l1_sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    from .jumprelu_l1_config import cfg, PROJECT
