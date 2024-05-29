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
from sae_components.core.collections.seq import ResidualSeq


def deep_resid_gated(
    d_data,
    d_dict,
    extra_layers=2,
    hidden_mult=2,
    mlp_mult=2,
    layer_nonlinearity=nn.LeakyReLU,
):
    d_hidden = d_data * hidden_mult

    b_dec = bias(d_data)

    @reused
    def deep_only():
        return Seq(
            pre_bias=Add(bias(d_data)),
            **(
                dict(project_up=MatMul(weight(d_data, d_hidden, scale=1)))
                if d_data != d_hidden
                else {}
            ),
            layers=ResidualSeq(
                *[
                    Seq(
                        mlp_layer(d_hidden, d_hidden * mlp_mult, scale=1),
                        nn.LayerNorm(d_hidden, device="cuda"),
                    )
                    for i in range(extra_layers)
                ],
            ),
            weight=MatMul(weight(d_hidden, d_dict)),
        )

    @reused
    def enc_gate():
        return Seq(
            deep=deep_only(),
            bias=ft.EncoderBias(bias(d_dict)).resampled(),
            nonlinearity=nn.ReLU(),
        )

    @reused
    def enc_mag():
        return Seq(
            deep=deep_only(),
            scaled=cl.ops.Parallel(
                identity=cl.ops.Identity(),
                r=bias(d_dict),
            ).reduce(
                lambda i, r: i * torch.exp(r),
            ),
            bias=ft.EncoderBias(bias(d_dict)).resampled(),
            nonlinearity=nn.ReLU(),
        )

    decoder = Seq(
        weight=ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                ft.DecoderWeights(MatMul(weight(d_dict, d_data))).resampled()
            )
        ),
        bias=cl.ops.Add(b_dec),
    )
    gated_model = Seq(
        encoder=cl.Parallel(
            magnitude=enc_mag(),
            gate=Thresh(enc_gate()),
        ).reduce(
            lambda x, y: x * y,
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        decoder=decoder,
    )

    model_aux = Seq(  # this one is just used for training the gate appropriately
        encoder=enc_gate(),  # oh and it's missing 1-2 detaches
        L1=L1Penalty(),
        freqs=EMAFreqTracker(),
        decoder=decoder,
    )
    losses = dict(
        L2_loss=L2Loss(gated_model),
        L2_aux_loss=L2Loss(model_aux),
        sparsity_loss=SparsityPenaltyLoss(model_aux),
    )
    return [gated_model], losses


def deep_resid_gated2(
    d_data,
    d_dict,
    extra_layers=2,
    hidden_mult=2,
    mlp_mult=2,
    layer_nonlinearity=nn.LeakyReLU,
):
    d_hidden = d_data * hidden_mult

    b_dec = bias(d_data)

    @reused
    def deep_only():
        return Seq(
            pre_bias_tied=cl.ops.Sub(b_dec),
            deep=Parallel(
                identity=cl.ops.Identity(),
                deep=Seq(
                    pre_bias=Add(bias(d_data)),
                    **(
                        dict(project_up=MatMul(weight(d_data, d_hidden, scale=1)))
                        if d_data != d_hidden
                        else {}
                    ),
                    layers=ResidualSeq(
                        *[
                            Seq(
                                mlp_layer(d_hidden, d_hidden * mlp_mult, scale=1),
                                nn.LayerNorm(d_hidden, device="cuda"),
                            )
                            for i in range(extra_layers)
                        ],
                    ),
                ),
            ).reduce(lambda i, d: torch.cat((i, d), dim=-1)),
            weight=MatMul(weight(d_hidden + d_data, d_dict)),
        )

    @reused
    def enc_gate():
        return Seq(
            deep=deep_only(),
            bias=ft.EncoderBias(bias(d_dict)).resampled(),
            nonlinearity=nn.ReLU(),
        )

    @reused
    def enc_mag():
        return Seq(
            deep=deep_only(),
            scaled=cl.ops.Parallel(
                identity=cl.ops.Identity(),
                r=bias(d_dict),
            ).reduce(
                lambda i, r: i * torch.exp(r),
            ),
            bias=ft.EncoderBias(bias(d_dict)).resampled(),
            nonlinearity=nn.ReLU(),
        )

    decoder = Seq(
        weight=ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                ft.DecoderWeights(MatMul(weight(d_dict, d_data))).resampled()
            )
        ),
        bias=cl.ops.Add(b_dec),
    )
    gated_model = Seq(
        encoder=cl.Parallel(
            magnitude=enc_mag(),
            gate=Thresh(enc_gate()),
        ).reduce(
            lambda x, y: x * y,
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        decoder=decoder,
    )

    model_aux = Seq(  # this one is just used for training the gate appropriately
        encoder=enc_gate(),  # oh and it's missing 1-2 detaches
        L1=L1Penalty(),
        freqs=EMAFreqTracker(),
        decoder=decoder,
    )
    losses = dict(
        L2_loss=L2Loss(gated_model),
        L2_aux_loss=L2Loss(model_aux),
        sparsity_loss=SparsityPenaltyLoss(model_aux),
    )
    return [gated_model, model_aux], losses
