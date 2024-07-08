import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

from saeco.architectures.initialization.initializer import Initializer
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
    Loss,
    L2Loss,
    SparsityPenaltyLoss,
    SAECache,
)

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Add, MatMul, Sub, Mul
from typing import Optional
from saeco.components.ops.fnlambda import Lambda
from saeco.core.reused_forward import ReuseForward, ReuseCache
from saeco.core import Seq
import saeco.components.features.features as ft
import saeco.components as co
from saeco.trainer.trainable import Trainable
from saeco.architectures.initialization.tools import bias, weight, mlp_layer


class SaveScale(cl.Module):
    def forward(self, x, *, cache: cl.Cache, **kwargs):
        cache.scale = ...
        with torch.no_grad():
            cache.scale = x
        return x


class Rescale(cl.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        y_pred = self.module(x, cache=cache, **kwargs)
        if cache._ancestor.has.scale and cache._ancestor.scale:
            with torch.no_grad():
                scale = x.norm(dim=1, keepdim=True) / y_pred.norm(dim=1, keepdim=True)
        else:
            return y_pred
        return y_pred * scale


def rescaling_vanilla(init: Initializer):
    init._decoder.bias = False
    b_dec = init.data_bias()
    model = Seq(
        pre_bias=Sub(b_dec),
        resc=Rescale(
            Seq(
                encoder=Seq(
                    lin=init.encoder,
                    nonlinearity=nn.ReLU(),
                ),
                freqs=EMAFreqTracker(),
                L1=L1Penalty(),
                metrics=co.metrics.ActMetrics(),
                decoder=ft.NormFeatures(
                    init.decoder,
                ),
            )
        ),
        post_bias=Add(b_dec),
    )

    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return models, losses
