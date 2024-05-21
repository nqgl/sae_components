import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

import sae_components.core as cl
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
from sae_components.core.linear import Bias, NegBias, Affine, MatMul
from typing import Optional

from sae_components.components.reused_forward import ReuseForward


class Detach: ...


class Mul: ...


class Thresh: ...


def gated_sae(d_data, d_dict):
    W_enc = ReuseForward(
        MatMul(nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict))))
    )
    W_dec = MatMul(nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data))))

    b_enc_mag = Bias(nn.Parameter(torch.zeros(d_dict)))

    b_enc_gate = Bias(nn.Parameter(torch.zeros(d_dict)))
    r_gate = Mul(nn.Parameter(torch.zeros(d_dict)))
    b_dec = Bias(nn.Parameter(torch.zeros(d_data)))
    b_dec_neg = ReuseForward(b_dec.tied_negative())

    enc_mag = cl.Seq(
        pre_bias=b_dec_neg,
        weight=W_enc,
        bias=b_enc_mag,
        nonlinearity=nn.ReLU(),
    )

    enc_gate = cl.Seq(
        pre_bias=b_dec_neg,
        weight=W_enc,
        r_mul=r_gate,
        bias=b_enc_gate,
        nonlinearity=nn.ReLU(),
    )

    enc_full = Parallel(enc_mag, Thresh(enc_gate)).set_reduction(lambda x, y: x * y)

    decoder = Affine(
        weight=W_dec,
        bias=b_dec,
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
    losses = [
        L2Loss(model),
        L2Loss(model_aux),
        SparsityPenaltyLoss(model_aux),
    ]
    return sae
