import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

import sae_components.core as cl
import sae_components.components as cc

from sae_components.components import (
    Penalty,
    L1Penalty,
    FreqTracked,
    EMAFreqTracker,
    FreqTracker,
    ResampledLayer,
    Loss,
    L2Loss,
    SparsityPenalty,
    SAECache,
)
from sae_components.core.linear import Bias, NegBias, Affine, MatMul
from typing import Optional


class SAE(cl.Sequential):
    encoder: cl.Module
    penalty: cl.Module
    decoder: cl.Module

    def __init__(self, encoder, penalty, decoder):
        super().__init__(
            encoder=encoder,
            penalty=penalty,
            decoder=decoder,
        )


class EncoderLayer(cl.Sequential):
    pre_bias: Optional[NegBias]
    affine: Affine
    nonlinearity: nn.ReLU

    def __init__(self, pre_bias: NegBias, affine: Affine, nonlinearity: nn.Module):
        if pre_bias is not None:
            super().__init__(
                pre_bias=pre_bias,
                affine=affine,
                nonlinearity=nonlinearity,
            )
        else:
            super().__init__(
                affine=affine,
                nonlinearity=nonlinearity,
            )

    @property
    def weight(self):
        return self.affine.weight

    @property
    def bias(self):
        return self.affine.bias

    # @property
    # def out_features(self):
    #     print("called out_features")
    #     print("self.affine", self.affine)
    #     print("2")
    #     return self.affine.out_features

    # @property
    # def in_features(self):
    #     return self.affine.in_features


class Resampled(cl.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, cache: SAECache, **kwargs):
        return self.module(x, cache=cache, **kwargs)


class Resampler:
    def __init__(
        self,
    ):
        self.enc = None
        self.dec = None
        self.b = None

    def encoder(self, enc: MatMul):
        assert self.enc is None
        self.enc = enc
        return enc

    def decoder(self, dec: MatMul):
        assert self.dec is None
        self.dec = dec
        return dec

    def bias(self, bias: Bias):
        assert self.b is None
        self.b = bias
        return bias


def vanilla_sae(d_data, d_dict):
    W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
    W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

    b_enc = Bias(nn.Parameter(torch.zeros(d_dict)))
    b_dec = Bias(nn.Parameter(torch.zeros(d_data)))

    resampler = Resampler()
    sae = cl.Sequential(
        encoder=cl.Sequential(
            EncoderLayer(
                pre_bias=b_dec.tied_negative(),
                affine=Affine(weight=W_enc, bias=b_enc),
                nonlinearity=nn.ReLU(),
            ),
            L1Penalty(),
        ),
        freqs=EMAFreqTracker(),
        decoder=Affine(
            weight=W_dec,
            bias=b_dec,
        ),
    )
    return sae


def main():
    d_data = 10
    d_dict = 20
    sae = vanilla_sae(d_data, d_dict).cuda()
    print(sae)
    x = torch.randn(8, d_data).cuda()
    cache = SAECache()
    y = sae(x, cache=cache)
    print(y)


if __name__ == "__main__":
    main()
