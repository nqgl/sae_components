import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

import sae_components.core as cl

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
    NegBias,
)


class SAE(nn.Module):
    def __init__(self, encoder, decoder, penalty):
        super().__init__()
        self.encoder = encoder
        self.penalty = penalty
        self.decoder = decoder

    def forward(self, x, cache: SAECache):
        encoded = self.encoder(x, cache=cache["encoder"])
        self.penalty(encoded, cache["encoder"])
        decoded = self.decoder(encoded, cache=cache["decoder"])
        return decoded


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
    pre_bias: NegBias
    affine: nn.Linear
    nonlinearity: nn.ReLU

    def __init__(self, pre_bias: NegBias, affine: nn.Linear, nonlinearity: nn.ReLU):
        super().__init__(
            pre_bias=pre_bias,
            affine=affine,
            nonlinearity=nonlinearity,
        )

    @property
    def weight(self):
        return self.affine.weight

    @property
    def bias(self):
        return self.affine.bias

    @property
    def out_features(self):
        print("called out_features")
        print("self.affine", self.affine)
        print("2")
        return self.affine.out_features

    @property
    def in_features(self):
        return self.affine.in_features


def vanilla_sae(d_data, d_dict):
    b_dec = nn.Parameter(torch.zeros(d_data))
    W_enc = nn.Parameter(torch.empty(d_data, d_dict))
    b_enc = nn.Parameter(torch.zeros(d_dict))
    W_dec = nn.Parameter(torch.empty(d_dict, d_data))

    sae = cl.Sequential(
        encoder=FreqTracked(
            module=EncoderLayer(
                pre_bias=NegBias(b_dec),
                affine=nn.Linear(d_data, d_dict),
                nonlinearity=nn.ReLU(),
            ),
        ),
        penalty=L1Penalty(),
        decoder=cl.CacheLayer(
            W=W_dec,
            bias=b_dec,
            nonlinearity=None,
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
