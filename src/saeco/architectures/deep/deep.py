import torch
import torch.nn as nn

from saeco.initializer.tools import bias
from saeco.initializer.tools import weight
from saeco.initializer.tools import mlp_layer
import saeco.core as cl
from saeco.core.collections.parallel import Parallel
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Add, MatMul
from saeco.core import Seq
import saeco.components.features.features as ft
import saeco.components as co


from saeco.core.collections.seq import ResidualSeq
from saeco.trainer.trainable import Trainable


def resid_deep_sae(
    d_data,
    d_dict,
    extra_layers=2,
    hidden_mult=2,
    mlp_mult=2,
    layer_nonlinearity=nn.PReLU,
):
    # parameters
    d_hidden = d_data * hidden_mult
    b_dec = bias(d_data)
    # model
    model = Seq(
        encoder=Seq(
            pre_bias=Add(bias(d_data)),
            **(
                dict(
                    project_up=MatMul(
                        nn.Parameter(
                            torch.eye(
                                d_data, device="cuda", dtype=torch.float32
                            ).repeat(1, 2)
                        )
                    )
                )
                if d_data != d_hidden
                else {}
            ),
            layers=ResidualSeq(
                *[
                    Seq(
                        mlp_layer(
                            d_hidden,
                            d_hidden * mlp_mult,
                            scale=1,
                            nonlinearity=layer_nonlinearity,
                        ),
                        nn.LayerNorm(d_hidden, device="cuda"),
                    )
                    for i in range(extra_layers)
                ],
            ),
            weight=MatMul(weight(d_hidden, d_dict, scale=2 ** (-0.5))),
            bias=ft.EncoderBias(bias(d_dict)).resampled(),
            nonlinearity=nn.ReLU(),
        ),
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

    # losses
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return [model], losses


def deep_sae(
    d_data, d_dict, extra_layers=1, hidden_mult=2, layer_nonlinearity=nn.LeakyReLU
):
    # parameters
    d_hidden = d_data * hidden_mult

    # model
    model = Seq(
        pre_bias=Add(bias(d_data)),
        encoder=Parallel(
            layers=Seq(
                *[
                    Seq(
                        weight=(
                            MatMul(weight(d_data, d_hidden))
                            if i == 0
                            else MatMul(weight(d_hidden, d_hidden))
                        ),
                        bias=Add(bias(d_hidden)),
                        nonlinearity=layer_nonlinearity(),
                    )
                    for i in range(extra_layers)
                ],
            ),
            identity=cl.ops.Identity(),
        ).reduce(lambda a, b: torch.cat((a, b), dim=-1)),
        weight=MatMul(weight(d_hidden + d_data, d_dict)),
        bias=ft.EncoderBias(bias(d_dict)).resampled(),
        nonlinearity=nn.ReLU(),
        freqs=EMAFreqTracker(),
        L1=L1Penalty(),
        metrics=co.metrics.ActMetrics(),
        decoder=Seq(
            weight=ft.OrthogonalizeFeatureGrads(
                ft.NormFeatures(
                    ft.DecoderWeights(weight(d_dict, d_data)).resampled(),
                )
            ),
            bias=cl.ops.Add(bias(d_data)),
        ),
    )

    # losses
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return [model], losses
