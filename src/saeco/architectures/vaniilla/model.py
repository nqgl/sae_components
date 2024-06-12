import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

import saeco.core as cl
import saeco.components as cc

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
import saeco.core.module
from saeco.core.linear import Bias, NegBias, Affine, MatMul
from typing import Optional


class SAE(cl.Seq):
    encoder: saeco.core.module.Module
    penalty: saeco.core.module.Module
    decoder: saeco.core.module.Module

    def __init__(self, encoder, penalty, decoder):
        super().__init__(
            encoder=encoder,
            penalty=penalty,
            decoder=decoder,
        )


class EncoderLayer(cl.Seq):
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


class Resampled(saeco.core.module.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *, cache: SAECache, **kwargs):
        return self.module(x, cache=cache, **kwargs)


class Resampler:
    def __init__(
        self,
    ):
        self.enc = None
        self.dec = None
        self.b = None
        self.freq_tracker: FreqTracker = EMAFreqTracker()

    def resampled_encoder(self, enc: MatMul):
        assert self.enc is None
        self.enc = enc
        return enc

    def resampled_decoder(self, dec: MatMul):
        assert self.dec is None
        self.dec = dec
        return dec

    def resampled_bias(self, bias: Bias):
        assert self.b is None
        self.b = bias
        return bias


def vanilla_sae(d_data, d_dict):
    W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
    W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

    b_enc = Bias(nn.Parameter(torch.zeros(d_dict)))
    b_dec = Bias(nn.Parameter(torch.zeros(d_data)))

    resampler = Resampler()
    sae = cl.Seq(
        encoder=cl.Seq(
            EncoderLayer(
                pre_bias=b_dec.tied_negative(),
                affine=Affine(
                    weight=W_enc,
                    bias=b_enc,
                ),
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


# def vanilla_sae(d_data, d_dict):
#     W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
#     W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

#     b_enc = nn.Parameter(torch.zeros(d_dict))
#     b_dec = nn.Parameter(torch.zeros(d_data))

#     resampler = Resampler()

#     sae = cl.Seq(
#         encoder=cl.Seq(
#             pre_bias=Bias(b_dec).tied_negative(),
#             weight=resampler.resampled_encoder(MatMul(W_enc)),
#             bias=resampler.resampled_bias(Bias(b_enc)),
#             nonlinearity=nn.ReLU(),
#         ),
#         penalty=L1Penalty(),
#         freq_tracker=resampler.freq_tracker,
#         decoder=cl.Seq(
#             weight=resampler.resampled_decoder(MatMul(W_dec)),
#             bias=Bias(b_dec),
#         ),
#     )
#     return sae


# def vanilla_sae(d_data, d_dict):
#     W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
#     W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

#     b_enc = Bias(nn.Parameter(torch.zeros(d_dict)))
#     b_dec = Bias(nn.Parameter(torch.zeros(d_data)))

#     resampler = Resampler()

#     sae = cl.Seq(
#         encoder=cl.Seq(
#             pre_bias=b_dec.tied_negative(),
#             weight=resampler.resampled_encoder(MatMul(W_enc)),
#             bias=resampler.resampled_bias(b_enc),
#             nonlinearity=nn.ReLU(),
#         ),
#         penalty=L1Penalty(),
#         freq_tracker=resampler.freq_tracker,
#         decoder=cl.Seq(
#             weight=resampler.resampled_decoder(MatMul(W_dec)),
#             bias=b_dec,
#         ),
#     )
#     return sae


# def vanilla_sae(d_data, d_dict):
#     W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_data, d_dict)))
#     W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_dict, d_data)))

#     b_enc = nn.Parameter(torch.zeros(d_dict))
#     b_dec = nn.Parameter(torch.zeros(d_data))

#     resampler = Resampler()

#     sae = cl.Seq(
#         -Bias(b_dec),
#         resampler.resampled_encoder(MatMul(W_enc)),
#         resampler.resampled_bias(Bias(b_enc)),
#         nn.ReLU(),
#         L1Penalty(),
#         resampler.freq_tracker,
#         resampler.resampled_decoder(MatMul(W_dec)),
#         Bias(b_dec),
#     )
#     return sae


def main():
    d_data = 10
    d_dict = 20
    sae = vanilla_sae(d_data, d_dict).cuda()
    print(sae)
    x = torch.randn(8, d_data).cuda()
    cache = SAECache()
    y = sae(x, cache=cache)
    print(y)
    print(sae.freqs.freqs)


if __name__ == "__main__":
    main()