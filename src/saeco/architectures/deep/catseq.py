import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float
from saeco.components.ops.fnlambda import Lambda

from saeco.architectures.initialization.tools import (
    reused,
    weight,
    bias,
    mlp_layer,
    layer,
)
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
)

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Add, MatMul, Sub, Mul
from typing import Optional
from saeco.core.reused_forward import ReuseCache
from saeco.core import Seq
import saeco.components.features.features as ft

import saeco.components as co
from saeco.core.collections.seq import ResidualSeq, CatSeq, CatSeqResid


def deep_catseq(
    d_data,
    d_dict,
    extra_layers=3,
    hidden_mult=3,
    mlp_mult=2,
    layer_nonlinearity=nn.GELU,
):
    d_hidden = d_data * hidden_mult

    b_dec = bias(d_data)

    @reused
    def encoder():
        return Seq(
            pre_bias=Sub(b_dec),
            layers=CatSeq(
                *[
                    layer(
                        d_data if i == 0 else d_hidden,
                        (d_hidden - d_data),
                        nonlinearity=layer_nonlinearity,
                        scale=0.1,
                    )
                    for i in range(extra_layers)
                ],
            ),
            weight=MatMul(weight(d_hidden, d_dict)),
            bias=Add(bias(d_dict)),
            nonlinearity=nn.ReLU(),
        )

    model = Seq(
        encoder=encoder(),
        freqs=EMAFreqTracker(),
        L1=L1Penalty(),
        metrics=co.metrics.ActMetrics(),
        decoder=Seq(
            weight=ft.OrthogonalizeFeatureGrads(
                ft.NormFeatures(
                    ft.DecoderWeights(weight(d_dict, d_data)).resampled(),
                )
            ),
            bias=cl.ops.Add(b_dec),
        ),
    )

    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )

    return models, losses


def deep_catseq_resid(
    d_data,
    d_dict,
    extra_layers=3,
    hidden_mult=3,
    mlp_mult=2,
    layer_nonlinearity=nn.LeakyReLU,
):
    d_hidden = d_data * hidden_mult

    b_dec = bias(d_data)

    @reused
    def encoder():
        return Seq(
            pre_bias=Sub(b_dec),
            layers=CatSeqResid(
                *[
                    Seq(
                        mlp_layer(
                            d_data if i == 0 else d_hidden,
                            d_hidden * mlp_mult,
                            (d_hidden - d_data),
                            nonlinearity=layer_nonlinearity,
                            scale=0.1,
                        ),
                        nn.LayerNorm(d_hidden - d_data, device="cuda"),
                    )
                    for i in range(extra_layers)
                ],
            ),
            weight=MatMul(weight(d_hidden, d_dict)),
            bias=Add(bias(d_dict)),
            nonlinearity=nn.ReLU(),
        )

    model = Seq(
        encoder=encoder(),
        freqs=EMAFreqTracker(),
        L1=L1Penalty(),
        metrics=co.metrics.ActMetrics(),
        decoder=Seq(
            weight=ft.OrthogonalizeFeatureGrads(
                ft.NormFeatures(
                    ft.DecoderWeights(weight(d_dict, d_data)).resampled(),
                )
            ),
            bias=cl.ops.Add(b_dec),
        ),
    )

    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )

    return models, losses
