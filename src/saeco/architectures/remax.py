import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

from saeco.architectures.tools import bias, weight, mlp_layer, Initializer
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
from saeco.components.ops.nonlinearities.softlu_remax import ReMax, ReMax1


def remax_sae(init: Initializer, remax1=False):
    scalenet = Seq(
        relu=nn.ReLU(),
        scale=nn.Linear(init.d_dict, 1),
        nonlinearity=Lambda(lambda x: torch.relu(x) + init.d_data**0.5),
    )

    model = Seq(
        encoder=Seq(
            pre_bias=Sub(init.decoder.bias),
            lin=init.encoder,
            nonlinearity=(ReMax1 if remax1 else ReMax)(scale=1),
            # scale=Mul(nn.Parameter(torch.ones(1))),
            scale=cl.ops.MulParallel(
                mag=scalenet,
                input=cl.ops.Identity(),
            ),
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


def remax1_sae(init: Initializer):
    return remax_sae(init, remax1=True)
