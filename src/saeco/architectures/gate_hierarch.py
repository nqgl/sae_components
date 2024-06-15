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
from saeco.components.gated import HGated, Gated


def hierarchical_l1scale(
    init: Initializer, num_levels=2, BF=2**5, detach=True, untied=True
):
    init._encoder.add_wrapper(ReuseForward)
    init._decoder.add_wrapper(ft.NormFeatures)
    init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
    # BF = 2**4

    def model(enc, penalties, metrics, detach=False):
        return Seq(
            pre_bias=ReuseForward(init._decoder.sub_bias()),
            **(
                dict(
                    detach=co.ops.Lambda(lambda x: x.detach()),
                )
                if detach
                else {}
            ),
            encoder=enc,
            freqs=EMAFreqTracker(),
            **penalties,
            metrics=metrics,
            decoder=init._decoder.detached if detach else init.decoder,
        )

    enc_mag = Seq(
        lin=init.encoder,
        nonlinearity=nn.ReLU(),
    )

    enc_gate = Seq(
        lin=init._encoder.make_new(),
        nonlinearity=nn.ReLU(),
    )
    gated = Gated(gate=enc_gate, mag=enc_mag)

    model_aux0 = model(
        gated.aux(),
        penalties=dict(l1=L1Penalty()),
        metrics=co.metrics.Metrics(L0=co.metrics.L0(), L1=co.metrics.L1()),
        detach=detach,
    )

    losses = {
        "L2_aux_loss0": L2Loss(model_aux0),
    }

    models = [model_aux0]
    encoders = [gated.full()]
    for l in range(1, num_levels + 1):
        bf = BF**l
        enc_hl = ReuseForward(
            Seq(
                lin=(init._encoder.make_hierarchical(bf=bf)),
                nonlinearity=nn.ReLU(),
            )
        )

        hgated = HGated(hl=enc_hl, ll=encoders[-1], bf=bf)

        hl_model_aux = model(
            hgated.aux(),
            penalties=dict(l1=L1Penalty(2.2**l)),
            metrics=co.metrics.Metrics(L0=co.metrics.L0(), L1=co.metrics.L1()),
            detach=detach,
        )
        models.append(hl_model_aux)
        encoders.append(hgated.full())
        losses[f"L2_aux_loss{l}"] = L2Loss(hl_model_aux)

    model_full = model(
        hgated.full(), penalties={}, metrics=co.metrics.ActMetrics(), detach=False
    )

    models = [model_full, *models]
    losses["L2_loss"] = L2Loss(model_full)
    losses["sparsity_loss"] = SparsityPenaltyLoss(
        model_full, num_expeted=num_levels + 1
    )

    return models, losses


# if __name__ == "__main__":
#     from saeco.trainer.runner import TrainingRunner
