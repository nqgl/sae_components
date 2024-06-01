import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float
from sae_components.components.ops.fnlambda import Lambda

from sae_components.architectures.tools import reused, weight, bias, mlp_layer, layer
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
)

# from sae_components.core.linear import Bias, NegBias, Affine, MatMul
from sae_components.core.basic_ops import Add, MatMul, Sub, Mul
from typing import Optional
from sae_components.core.reused_forward import ReuseCache
from sae_components.core import Seq
import sae_components.components.decoder_normalization.features as ft

import sae_components.components as co
from sae_components.core.collections.seq import ResidualSeq, CatSeq, CatSeqResid


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
