import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

from saeco.architectures.base import SAE
from saeco.architectures.initialization.initializer import Initializer
import saeco.core.module
from saeco.core.collections.parallel import Parallel
from saeco.components import (
    Penalty,
    L1Penalty,
    FreqTracked,
    EMAFreqTracker,
    FreqTracker,
    ResampledLayer,
    Loss,
    L2Loss,
    SparsityPenaltyLoss,
    SAECache,
)

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Add, MatMul, Sub, Mul
from typing import Optional
from saeco.core.reused_forward import ReuseForward, ReuseCache
from saeco.core import Seq
import saeco.components.features.features as ft
from saeco.architectures.initialization.tools import (
    reused,
    weight,
)

import saeco.components as co
from saeco.trainer.trainable import Trainable
import einops
from saeco.components.gated import HGated, Gated, ClassicGated
from saeco.misc import useif
import saeco.core as cl
from saeco.components.penalties.l1_penalizer import L0TargetingL1Penalty


def hierarchical_l1scale(
    init: Initializer,
    num_levels=5,
    BF=2**2,
    detach=False,
    untied=True,
    pre_bias=False,
    classic=False,
    full_mode=False,
    soft=False,
):
    init._encoder.add_wrapper(ReuseForward)
    init._decoder.add_wrapper(ft.NormFeatures)
    init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
    if classic:
        init._encoder.bias = False
    # BF = 2**4
    detached_pre_bias = cl.Parallel(
        left=cl.ops.Identity(), right=init.decoder.bias
    ).reduce((lambda l, r: l - r.detach()))
    l1_penalties = [L1Penalty()]

    def model(enc, penalties, metrics, detach=False):
        return Seq(
            **useif(
                pre_bias,
                pre_bias=(
                    ReuseForward(detached_pre_bias)
                    if detach
                    else cl.ReuseForward(init._decoder.sub_bias())
                ),
            ),
            encoder=enc,
            freqs=EMAFreqTracker(),
            **penalties,
            metrics=metrics,
            decoder=init._decoder.detached if detach else init.decoder,
        )

    if classic:
        gated = ClassicGated(init=init)
    else:
        enc_mag = Seq(
            lin=init.encoder,
            nonlinearity=nn.ReLU(),
        )

        enc_gate = Seq(
            lin=init._encoder.make_new(),
            nonlinearity=nn.ReLU(),
        )
        gated = Gated(gate=enc_gate, mag=enc_mag)
        # gated = HGated(hl=enc_gate, ll=enc_mag, bf=1)

    layer = gated
    losses = {}
    L1_SCALE_BASE = 1.5
    for l in range(1, num_levels + 1):
        bf = BF**l
        l1_penalties.append(
            L0TargetingL1Penalty(init.d_dict // bf / 4, L1_SCALE_BASE**l)
        )
        enc_hl = ReuseForward(
            Seq(
                lin=(init._encoder.make_hierarchical(bf=bf)),
                nonlinearity=nn.ReLU(),
                read_l0_for_l1=co.Lambda(l1_penalties[-1].update_l0),
            )
        )

        layer = HGated(hl=enc_hl, ll=layer, bf=bf)
    models = []
    for l in range(num_levels + 1):
        aux_model = model(
            enc=layer.aux(num_levels - l, full_mode=full_mode, soft=soft),
            penalties=dict(l1=l1_penalties[l]),
            metrics=co.metrics.Metrics(L0=co.metrics.L0(), L1=co.metrics.L1()),
            detach=detach,
        )
        losses[f"L2_aux_loss{l}"] = L2Loss(aux_model)
        models.append(aux_model)

    model_full = model(
        layer.full(), penalties={}, metrics=co.metrics.ActMetrics(), detach=False
    )

    models = [model_full, *models]
    losses["L2_loss"] = L2Loss(model_full)
    losses["sparsity_loss"] = SparsityPenaltyLoss(
        model_full, num_expeted=num_levels + 1
    )

    return models, losses


# if __name__ == "__main__":
#     from saeco.trainer.runner import TrainingRunner
