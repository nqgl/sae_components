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


def lprint(x):
    def l(i):
        print(x)
        return i

    return Lambda(cl.ops.Identity(), l)


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
    mm_W_enc = ReuseForward(MatMul(W_enc))

    # encoders
    enc_mag = Seq(
        pre_bias=sub_enc_shared_pre_bias,
        weight=mm_W_enc,
        bias=Add(b_enc_mag),
        nonlinearity=nn.ReLU(),
    )

    enc_gate = Seq(
        pre_bias=sub_enc_shared_pre_bias,
        weight=mm_W_enc,
        r_mul=Mul(r_gate),
        bias=Add(b_enc_gate),
        nonlinearity=nn.ReLU(),
    )

    # decoder
    decoder = ReuseForward(
        Seq(
            weight=MatMul(W_dec),
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
        l1=L1Penalty(),
        freqs=EMAFreqTracker(),
        decoder=decoder,
    )

    # losses
    losses = cl.Parallel(
        L2=L2Loss(gated_model),
        L2_aux=L2Loss(model_aux),
        sparsisty=SparsityPenaltyLoss(model_aux),
    ).reduce(torch.sum)
    return gated_model, losses


def main():
    d_data = 100
    d_dict = 50
    model, losses = gated_sae(d_data, d_dict)
    print(model)
    print(losses)

    print(model.state_dict())

    x = torch.randn(7, d_data)
    cache = SAECache()
    cache += ReuseCache()
    y = model(x, cache=cache)

    print(y)


if __name__ == "__main__":
    main()
