import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

from sae_components.architectures.tools import bias, weight, mlp_layer, Initializer
from sae_components.components.ops.detach import Thresh
import sae_components.core as cl
import sae_components.core.module
from sae_components.core.collections.parallel import Parallel
from sae_components.components import (
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

# from sae_components.core.linear import Bias, NegBias, Affine, MatMul
from sae_components.core.basic_ops import Add, MatMul, Sub, Mul
from typing import Optional
from sae_components.components.ops.fnlambda import Lambda
from sae_components.core.reused_forward import ReuseForward, ReuseCache
from sae_components.core import Seq
import sae_components.components.features.features as ft
import sae_components.components as co


from sae_components.core.collections.seq import ResidualSeq
from sae_components.trainer.trainable import Trainable


def TopK(k):
    def _topk(x):
        v, i = x.topk(k, dim=-1, sorted=False)
        return torch.zeros_like(x).scatter_(-1, i, v)

    return Lambda(_topk)


def topk_sae(init: Initializer, cfg={"k": 45}):
    model = Seq(
        encoder=Seq(
            pre_bias=Sub(init.decoder.bias),
            lin=init.encoder,
            nonlinearity=nn.ReLU(),
            topk=TopK(cfg["k"]),
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
