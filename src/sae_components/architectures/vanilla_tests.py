import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

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
import sae_components.components.features.features as ft
import sae_components.components as co
from sae_components.trainer.trainable import Trainable


# from torch.utils.viz._cycles import
def lprint(x):
    def l(i):
        print(x)
        return i

    return Lambda(l)


def vanilla_sae(d_data, d_dict):
    # parameters
    W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
    W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

    b_enc = nn.Parameter(torch.zeros(d_dict))
    b_dec = nn.Parameter(torch.zeros(d_data))

    # model
    model = Seq(
        encoder=Seq(
            pre_bias=Sub(b_dec),
            weight=ft.EncoderWeights(W_enc).resampled(),
            bias=ft.EncoderBias(b_enc).resampled(),
            nonlinearity=nn.ReLU(),
        ),
        freqs=EMAFreqTracker(),
        L1=L1Penalty(),
        metrics=co.metrics.ActMetrics(),
        decoder=Seq(
            weight=ft.OrthogonalizeFeatureGrads(
                ft.NormFeatures(
                    ft.DecoderWeights(W_dec).resampled(),
                )
            ),
            bias=cl.ops.Add(b_dec),
        ),
    )

    # losses
    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return models, losses


def basic_vanilla_sae(d_data, d_dict):
    # parameters
    W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
    W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

    b_enc = nn.Parameter(torch.zeros(d_dict))
    b_dec = nn.Parameter(torch.zeros(d_data))

    # model
    model = Seq(
        encoder=Seq(
            pre_bias=Sub(b_dec),
            weight=ft.EncoderWeights(W_enc).resampled(),
            bias=ft.EncoderBias(b_enc).resampled(),
            nonlinearity=nn.ReLU(),
        ),
        freqs=EMAFreqTracker(),
        L1=L1Penalty(),
        metrics=co.metrics.ActMetrics(),
        decoder=Seq(
            weight=ft.DecoderWeights(W_dec).resampled(),
            bias=cl.ops.Add(b_dec),
        ),
    )

    # losses
    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return models, losses


def basic_vanilla_sae_lin(d_data, d_dict):
    # parameters
    W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
    W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

    b_enc = nn.Parameter(torch.zeros(d_dict))
    b_dec = nn.Parameter(torch.zeros(d_data))
    dec = co.LinDecoder(nn.Linear(d_dict, d_data))
    enc = co.LinEncoder(nn.Linear(d_data, d_dict))
    # enc.weight.data[:] = W_enc.transpose(-1, -2)
    # dec.weight.data[:] = W_dec.transpose(-1, -2)
    enc.weight.data[:] = enc.weight * 3**0.5
    dec.weight.data[:] = dec.weight * 3**0.5
    # nn.init.kaiming_uniform_(enc.weight.data)
    # nn.init.kaiming_uniform_(dec.weight.data)
    # dec.bias.data[:] = 0
    # enc.bias.data[:] = 0

    # dec.weight.data[:]

    # model
    model = Seq(
        encoder=Seq(
            pre_bias=Sub(dec.bias),
            lin=enc,
            nonlinearity=nn.ReLU(),
        ),
        freqs=EMAFreqTracker(),
        L1=L1Penalty(),
        metrics=co.metrics.ActMetrics(),
        decoder=ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                dec,
            ),
        ),
    )

    # losses
    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return models, losses


def basic_vanilla_sae_lin_no_orth(d_data, d_dict):
    # parameters
    W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
    W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

    b_enc = nn.Parameter(torch.zeros(d_dict))
    b_dec = nn.Parameter(torch.zeros(d_data))
    dec = co.LinDecoder(nn.Linear(d_dict, d_data))
    enc = co.LinEncoder(nn.Linear(d_data, d_dict))
    # enc.weight.data[:] = W_enc.transpose(-1, -2)
    # dec.weight.data[:] = W_dec.transpose(-1, -2)
    enc.weight.data[:] = enc.weight * 3**0.5
    dec.weight.data[:] = dec.weight * 3**0.5
    # nn.init.kaiming_uniform_(enc.weight.data)
    # nn.init.kaiming_uniform_(dec.weight.data)
    # dec.bias.data[:] = 0
    # enc.bias.data[:] = 0

    # dec.weight.data[:]

    # model
    model = Seq(
        encoder=Seq(
            pre_bias=Sub(dec.bias),
            lin=enc,
            nonlinearity=nn.ReLU(),
        ),
        freqs=EMAFreqTracker(),
        L1=L1Penalty(),
        metrics=co.metrics.ActMetrics(),
        decoder=ft.NormFeatures(
            dec,
        ),
    )

    # losses
    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return models, losses


# def basic_vanilla_sae_lin_test_nonfused(d_data, d_dict):
#     # parameters
#     W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
#     W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

#     b_enc = nn.Parameter(torch.zeros(d_dict))
#     b_dec = nn.Parameter(torch.zeros(d_data))
#     dec = nn.Linear(d_dict, d_data, bias=False).cuda()
#     # model
#     model = Seq(
#         encoder=Seq(
#             pre_bias=Sub(b_dec),
#             lin=nn.Linear(d_data, d_dict, bias=False).cuda(),
#             bias=Add(b_enc),
#             nonlinearity=nn.ReLU(),
#         ),
#         freqs=EMAFreqTracker(),
#         L1=L1Penalty(),
#         metrics=co.metrics.ActMetrics(),
#         decoder=dec,
#         bias=Add(b_dec),
#     )

#     # losses
#     models = [model]
#     losses = dict(
#         L2_loss=L2Loss(model),
#         sparsity_loss=SparsityPenaltyLoss(model),
#     )
#     return models, losses


# def vanilla_sae(d_data, d_dict):
#     # parameters
#     W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
#     W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

#     b_enc = nn.Parameter(torch.zeros(d_dict))
#     b_dec = nn.Parameter(torch.zeros(d_data))

#     # model
#     model = Seq(
#         encoder=Seq(
#             pre_bias=Sub(b_dec),
#             weight=ft.EncoderWeights(W_enc),
#             bias=ft.EncoderBias(b_enc),
#             nonlinearity=nn.ReLU(),
#         ),
#         freqs=EMAFreqTracker(),
#         L1=L1Penalty(),
#         decoder=Seq(
#             weight=ft.DecoderWeights(W_dec),
#             bias=cl.ops.Add(b_dec),
#         ),
#     )

#     # losses
#     losses = dict(
#         L2_loss=L2Loss(model),
#         sparsisty_loss=SparsityPenaltyLoss(model),
#     )
#     return model, losses


d_data = 768
d_dict = 8 * d_data


def test_train(model, losses):
    features = torch.randn(d_dict, d_data).cuda()
    from sae_components.trainer.trainer import Trainer
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


def main():
    sae = vanilla_sae
    model, losses = sae(d_data, d_dict)
    test_train(model, losses)


if __name__ == "__main__":
    main()
