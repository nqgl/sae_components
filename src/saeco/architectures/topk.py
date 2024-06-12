import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

from saeco.architectures.initialization.initializer import Initializer
from saeco.architectures.initialization.tools import bias, weight, mlp_layer
from saeco.components.ops.detach import Thresh
import saeco.core as cl
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
    LambdaPenalty,
)

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Add, MatMul, Sub, Mul
from typing import Optional
from saeco.components.ops.fnlambda import Lambda
from saeco.core.reused_forward import ReuseForward, ReuseCache
from saeco.core import Seq
import saeco.components.features.features as ft
import saeco.components as co


from saeco.core.collections.seq import ResidualSeq
from saeco.trainer.trainable import Trainable


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
        null_penalty=LambdaPenalty(lambda x: 0),  # "no sparsity penalty"
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
