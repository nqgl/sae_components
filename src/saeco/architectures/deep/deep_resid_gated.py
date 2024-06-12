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
from saeco.core.collections.seq import ResidualSeq


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
                        dict(
                            project_up=MatMul(weight(d_data, d_hidden, scale=1)),
                            proj_ln=nn.LayerNorm(d_hidden, device="cuda"),
                        )
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


def deep_resid_gated2_deeper(
    d_data,
    d_dict,
):
    return deep_resid_gated2(d_data, d_dict, extra_layers=4, hidden_mult=3, mlp_mult=2)


def deep_resid_gated2_deeper_still(
    d_data,
    d_dict,
):
    return deep_resid_gated2(d_data, d_dict, extra_layers=6, hidden_mult=3, mlp_mult=4)


def deep_resid_gated2_wider(
    d_data,
    d_dict,
):
    return deep_resid_gated2(d_data, d_dict, extra_layers=2, hidden_mult=6, mlp_mult=3)


def deep_resid_gated2_wider2(d_data, d_dict):
    return deep_resid_gated2(d_data, d_dict, extra_layers=1, hidden_mult=8, mlp_mult=4)
