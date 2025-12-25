import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
from saeco.components import (
    EMAFreqTracker,
    L1Penalty,
    L2Loss,
    SparsityPenaltyLoss,
)
from saeco.components.ops.fnlambda import Lambda
from saeco.core import Seq

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Sub
from saeco.initializer import Initializer


def lprint(x):
    def l(i):
        print(x)
        return i

    return Lambda(l)


def basic_vanilla_sae_lin(init: Initializer):
    # model
    model = Seq(
        encoder=Seq(
            pre_bias=Sub(init.decoder.bias),
            lin=init.encoder,
            nonlinearity=nn.ReLU(),
        ),
        freqs=EMAFreqTracker(),
        L1=L1Penalty(),
        metrics=co.metrics.ActMetrics(),
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


def classed_basic_vanilla_sae_lin(init: Initializer):
    # model
    from saeco.architectures.base import SAE

    model = SAE(
        pre_bias=Sub(init.decoder.bias),
        encoder=Seq(
            lin=init.encoder,
            nonlinearity=nn.ReLU(),
        ),
        penalty=L1Penalty(),
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
