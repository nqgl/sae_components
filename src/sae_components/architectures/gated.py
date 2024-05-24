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
from sae_components.components.ops import Lambda
from sae_components.core.reused_forward import ReuseForward, ReuseCache
from sae_components.core import Seq
import sae_components.components.decoder_normalization.features as ft


def gated_sae(d_data, d_dict):
    # parameters
    W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
    W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

    b_enc_mag = nn.Parameter(torch.zeros(d_dict))

    b_enc_gate = nn.Parameter(torch.zeros(d_dict))
    r_gate = nn.Parameter(torch.ones(d_dict))

    b_dec = nn.Parameter(torch.zeros(d_data))

    # shared components
    sub_enc_shared_pre_bias = ReuseForward(Sub(b_dec))
    mm_W_enc = ReuseForward(ft.EncoderWeights(W_enc).resampled())

    # encoders
    enc_mag = Seq(
        pre_bias=sub_enc_shared_pre_bias,
        weight=mm_W_enc,
        bias=ft.EncoderBias(b_enc_mag).resampled(),
        nonlinearity=nn.ReLU(),
    )

    enc_gate = Seq(
        pre_bias=sub_enc_shared_pre_bias,
        weight=mm_W_enc,
        r_mul=Mul(r_gate),
        bias=ft.EncoderBias(b_enc_gate).resampled(),
        nonlinearity=nn.ReLU(),
    )

    # decoder
    decoder = ReuseForward(
        Seq(
            weight=ft.OrthogonalizeFeatureGrads(
                ft.NormFeatures(
                    ft.DecoderWeights(W_dec).resampled(),
                )
            ),
            bias=cl.ops.Add(b_dec),
        )
    )

    # models
    gated_model = Seq(
        encoder=cl.Parallel(
            magnitude=enc_mag,
            gate=Thresh(enc_gate),
        ).reduce(
            lambda x, y: x * y,
        ),
        freqs=EMAFreqTracker(),
        decoder=decoder,
    )

    model_aux = Seq(  # this one is just used for training the gate appropriately
        encoder=enc_gate,  # oh and it's missing 1-2 detaches
        L1=L1Penalty(),
        # L0_aux=Lambda(cl.ops.Identity(), lambda x: (x > 0).sum(0).float().mean()),
        freqs=EMAFreqTracker(),
        decoder=decoder,
    )

    # losses
    losses = dict(
        L2_loss=L2Loss(gated_model),
        L2_aux_loss=L2Loss(model_aux),
        sparsisty_loss=SparsityPenaltyLoss(model_aux),
    )
    return gated_model, losses


def main():
    d_data = 1000 * 5
    d_dict = 500 * 5
    model, losses = gated_sae(d_data, d_dict)
    model = model.cuda()
    print(model)
    print(losses)

    print(model.state_dict())

    x = torch.randn(7000, d_data).cuda()
    cache = SAECache()
    cache += ReuseCache()
    y = model(x, cache=cache)

    print(y)


def test_train(model, losses, data):
    d_data = 768
    d_dict = 8 * d_data
    features = torch.randn(d_dict, d_data).cuda()
    from sae_components.trainer.trainer import Trainer, Trainable
    import tqdm
    import wandb

    trainer = Trainer({}, Trainable([model], losses).cuda())
    batch_size = 4096

    trainer.train(data)


def main():
    d_data = 768
    d_dict = 8 * d_data
    features = torch.randn(d_dict, d_data).cuda()
    model, losses = gated_sae(d_data, d_dict)
    from sae_components.trainer.trainer import Trainer, Trainable
    import tqdm
    import wandb

    trainer = Trainer({}, Trainable([model], losses).cuda())
    batch_size = 4096

    @torch.no_grad()
    def data_generator():
        rand = torch.rand(batch_size, d_dict, device="cuda")
        for i in tqdm.trange(10000):
            rand[:] = rand + 0.001
            x = rand @ features
            yield x

    # for i in data_generator():
    #     pass

    trainer.train(data_generator())


if __name__ == "__main__":
    main()
