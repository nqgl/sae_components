import torch

import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl

from saeco.architectures.initialization.initializer import Initializer
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


class AnthUpdateConfig(SweepableConfig):
    pre_bias: bool = False
    clip_grad: float | None = 1


def anth_update_model(
    init: Initializer,
    cfg: AnthUpdateConfig,
):
    dec_mul_l1 = L1PenaltyScaledByDecoderNorm()
    init._decoder.const_init_bias(0)
    init._encoder.const_init_bias(0)
    model_full = Seq(
        **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
        encoder=Seq(linear=init.encoder.resampled(), relu=nn.ReLU()),
        freqs=EMAFreqTracker(),
        l1=dec_mul_l1,
        metrics=co.metrics.ActMetrics(),
        decoder=dec_mul_l1.set_decoder(init.decoder.resampled()),
    )
    if cfg.clip_grad:
        model_full = ClipGrad(model_full)
    models = [model_full]
    losses = dict(
        L2_loss=L2Loss(model_full),
        sparsity_loss=SparsityPenaltyLoss(model_full),
    )

    return models, losses
