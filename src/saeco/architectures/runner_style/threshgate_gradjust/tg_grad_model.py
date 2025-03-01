import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
import torch

import torch.nn as nn
from saeco.components import EMAFreqTracker, L1Penalty, L2Loss, SparsityPenaltyLoss

from saeco.components.hooks.clipgrad import ClipGrad
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.core import Seq

from saeco.initializer import Initializer

from saeco.misc import useif

from .threshgrad import BinaryEncoder, Config, GTTest


def tg_grad_sae(
    init: Initializer,
    cfg: Config,
):
    model = Seq(
        encoder=Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            lin=GTTest(cfg, init) if cfg.mag_weights else BinaryEncoder(cfg, init),
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        penalty=co.LinearDecayL1Penalty(40_000) if cfg.decay_l1 else co.L1Penalty(),
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
    tr = TrainingRunner(cfg, model_fn=tg_grad_sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    from .tg_grad_config import cfg, PROJECT
