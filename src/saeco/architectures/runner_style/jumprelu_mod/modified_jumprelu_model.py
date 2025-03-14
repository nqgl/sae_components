import torch

import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.components.jumprelu.jumprelu_fn2
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
from saeco.sweeps import SweepableConfig
from saeco.components.jumprelu.jumprelu_fn import HStep, JumpReLU, L0Penalty
from saeco.components.jumprelu.kernels_fns import kernels
from saeco.components.features.param_metadata import ParamMetadata
import saeco.components.jumprelu.jumprelu_fn as jfun
from saeco.components.jumprelu.jumprelu_fn2 import modify_modified

modify_modified()
#
# jfun.jumprelu_modified = saeco.components.jumprelu.jumprelu_fn2.jumprelu_modified2


class Config(SweepableConfig):
    eps: float
    kernel: str = "rect"
    pre_bias: bool = False
    thresh_initial_value: float = (
        0.5  # 1 better than 0 definitely, but seems 0.5 maybe better than 1
    )
    thresh_lr_mult: float = 1.0
    # depends prob on lots of variables
    # normalization obv, also L0 target, initialization magnitudes
    # probably other stuff
    modified_jumprelu_grad: int = False
    modified_thresh_grad: int = False
    penalize_pre_acts: bool = True
    exp: bool = False
    leniency: float = 1.0

    def get_kernel(self):
        return kernels[self.kernel]


def jumprelu_sae(
    init: Initializer,
    cfg: Config,
):
    init.new_encoder_bias()
    thresh = nn.Parameter(
        torch.zeros(init.d_dict) + cfg.thresh_initial_value
    )  # TODO resample (reset on resample) thresh
    if cfg.exp:
        with torch.no_grad():
            thresh = thresh.log_()
    thresh_metadata = ParamMetadata(thresh)
    thresh_metadata.lr_mult(cfg.thresh_lr_mult)
    model = Seq(
        encoder=Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            lin=init.encoder,
            **useif(
                cfg.penalize_pre_acts,
                penalty=L0Penalty(
                    thresh=thresh,
                    kernel=cfg.get_kernel(),
                    eps=cfg.eps,
                    exp=cfg.exp,
                    modified_grad=(cfg.modified_thresh_grad, cfg.leniency, 768),
                ),
            ),
            nonlinearity=JumpReLU(
                thresh=thresh,
                kernel=cfg.get_kernel(),
                eps=cfg.eps,
                exp=cfg.exp,
                modified_jumprelu=(cfg.modified_jumprelu_grad, cfg.leniency, 768),
            ),
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        **useif(
            not cfg.penalize_pre_acts,
            penalty=L0Penalty(
                thresh=thresh,
                kernel=cfg.get_kernel(),
                eps=cfg.eps,
                exp=cfg.exp,
                modified_grad=(cfg.modified_thresh_grad, cfg.leniency, 768),
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
    tr = TrainingRunner(cfg, model_fn=jumprelu_sae)
    # pgs = tr.trainable.param_groups({"lr": 1})
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    from .jumprelu_config import cfg, PROJECT
