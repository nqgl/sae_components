import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

from sae_components.architectures.tools import bias
from sae_components.architectures.tools import weight
from sae_components.architectures.tools import mlp_layer
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
from sae_components.components.ops.fnlambda import Lambda
from sae_components.core.reused_forward import ReuseForward, ReuseCache
from sae_components.core import Seq
import sae_components.components.decoder_normalization.features as ft
import sae_components.components as co


from sae_components.core.collections.seq import ResidualSeq


def resid_deep_sae(
    d_data,
    d_dict,
    extra_layers=2,
    hidden_mult=2,
    mlp_mult=2,
    layer_nonlinearity=nn.LeakyReLU,
):
    # parameters
    d_hidden = d_data * hidden_mult
    b_dec = bias(d_data)
    # model
    model = Seq(
        encoder=Seq(
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
    return model, losses


def deep_sae(
    d_data, d_dict, extra_layers=1, hidden_mult=2, layer_nonlinearity=nn.LeakyReLU
):
    # parameters
    d_hidden = d_data * hidden_mult

    # model
    model = Seq(
        encoder=Seq(
            pre_bias=Add(bias(d_data)),
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
            weight=MatMul(weight(d_hidden, d_dict)),
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
            bias=cl.ops.Add(bias(d_data)),
        ),
    )

    # losses
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return model, losses


d_data = 768
d_dict = 8 * d_data


def test_train(model, losses):
    features = torch.randn(d_dict, d_data).cuda()
    from sae_components.trainer.trainer import Trainer, Trainable
    import tqdm
    import wandb

    trainer = Trainer({}, Trainable([model], losses).cuda())
    batch_size = 4096 * 4

    @torch.no_grad()
    def data_generator():
        for i in tqdm.trange(10000):
            rand = torch.rand(batch_size, d_dict, device="cuda")
            x = rand @ features
            yield x

    trainer.train(data_generator())
