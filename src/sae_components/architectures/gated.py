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


def lprint(x):
    def l(i):
        print(x)
        return i

    return Lambda(cl.ops.Identity(), l)


def gated_sae(d_data, d_dict):
    mm_W_enc = ReuseForward(
        MatMul(
            nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict))),
        )
    )
    mm_W_dec = ReuseForward(
        MatMul(nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data))))
    )

    b_enc_mag = nn.Parameter(torch.zeros(d_dict))
    b_enc_gate = nn.Parameter(torch.zeros(d_dict))

    r_gate = nn.Parameter(torch.ones(d_dict))
    b_dec = nn.Parameter(torch.zeros(d_data))

    sub_enc_shared_pre_bias = ReuseForward(Sub(b_dec))

    enc_mag = cl.Seq(
        pre_bias=sub_enc_shared_pre_bias,
        weight=mm_W_enc,
        bias=Add(b_enc_mag),
        nonlinearity=nn.ReLU(),
    )

    enc_gate = cl.Seq(
        pre_bias=sub_enc_shared_pre_bias,
        weight=mm_W_enc,
        r_mul=Mul(r_gate),
        bias=Add(b_enc_gate),
        nonlinearity=nn.ReLU(),
    )
    enc_full = cl.ops.Parallel(
        magnitude=enc_mag,
        gate=Thresh(enc_gate),
    ).reduce(
        lambda x, y: x * y,
    )

    decoder = cl.Seq(
        weight=mm_W_dec,
        bias=cl.ops.Add(b_dec),
    )

    # resampler = Resampler()
    model = cl.Seq(
        encoder=enc_full,
        freqs=EMAFreqTracker(),
        decoder=decoder,
    )
    model_aux = cl.Seq(
        encoder=enc_gate,
        l1=L1Penalty(),
        freqs=EMAFreqTracker(),
        decoder=decoder,
    )
    losses = cl.ops.AddParallel(
        L2=L2Loss(model),
        L2_aux=L2Loss(model_aux),
        sparsisty=SparsityPenaltyLoss(model_aux),
    )
    return model, losses


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


# def main():
#     import tqdm

#     mmm = MatMul(nn.Parameter(torch.randn(768, 768 * 16)))
#     rmmm = ReuseForward(mmm)
#     model = cl.Seq(
#         mm1=rmmm,
#         mm2=MatMul(nn.Parameter(torch.randn(768 * 16, 768))),
#     ).cuda()

#     print(model)
#     o = torch.optim.Adam(model.parameters())
#     x = torch.randn(2 * 4096, 768).cuda()
#     for i in tqdm.tqdm(range(1000)):
#         x = x - 0.01
#         cache = SAECache()
#         cache += ReuseCache()
#         cache.input = ...
#         y = model(x, cache=cache)
#         loss = (100 - y.sum()) ** 2
#         loss.backward()
#         o.step()
#         o.zero_grad()
#     print(cache._subcaches)


if __name__ == "__main__":
    main()
