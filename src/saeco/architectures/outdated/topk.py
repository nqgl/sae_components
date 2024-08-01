import torch
import torch.nn as nn

from saeco.architectures.initialization.initializer import Initializer
from saeco.components import (
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
    LambdaPenalty,
)

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Sub
from saeco.components.ops.fnlambda import Lambda
from saeco.core import Seq
import saeco.components.features.features as ft
import saeco.components as co


def TopK(k):
    def _topk(x):
        v, i = x.topk(k, dim=-1, sorted=False)
        return torch.zeros_like(x).scatter_(-1, i, v)

    return Lambda(_topk)


def topk_sae(init: Initializer):
    model = Seq(
        encoder=Seq(
            pre_bias=Sub(init.decoder.bias),
            lin=init.encoder,
            nonlinearity=nn.ReLU(),
            topk=TopK(init.l0_target),
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        null_penalty=LambdaPenalty(lambda x: torch.zeros(1)),  # "no sparsity penalty"
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
