import torch
from torch import Tensor

import saeco.core as cl

from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    SAECache,
)
from typing import Protocol
import saeco.components as co


class LinearLike(Protocol):
    weight: Tensor
    bias: Tensor


class SAE(cl.Seq):
    encoder: cl.Module
    penalty: cl.Module
    decoder: cl.Module
    metrics: co.metrics.ActMetrics
    freq_tracker: co.FreqTracker

    def __init__(
        self,
        pre_bias,
        encoder,
        decoder,
        b_dec=None,
        penalty=True,
        metrics=None,
        freq_tracker=None,
    ):
        self._b_dec = b_dec
        super().__init__(
            **(
                {}
                if pre_bias is False
                else {"pre_bias": pre_bias or cl.ops.Sub(self.b_dec)}
            ),
            encoder=encoder,
            **(
                {}
                if penalty is None
                else dict(penalty=(L1Penalty() if penalty is True else penalty))
            ),
            freq_tracker=freq_tracker or EMAFreqTracker(),
            metrics=metrics or co.metrics.ActMetrics(),
            decoder=decoder,
        )

    @property
    def b_dec(self):
        if self._b_dec is None:
            return self.decoder.bias
        return self._b_dec

    @property
    def freqs(self): ...

    def encode(self, x):
        if hasattr(self, "pre_bias"):
            x = self.pre_bias(x)
        return self.encoder(x)

    def decode(self, x):
        if self._b_dec is None:
            return self.decoder(x)
        else:
            return self.decoder(x) + self.b_dec

    def cent(self, x):
        if hasattr(self, "pre_bias"):
            return self.pre_bias(x)
        raise NotImplementedError

    def uncent(self, x):
        return x + self.b_dec

    def enc(self, x, *, cache: SAECache = None):
        if cache is None:
            cache = SAECache()
        return cache(self).encoder(x)
        # return self.encoder(x, cache=cache)

    def dec(self, x, *, cache: SAECache = None):
        if cache is None:
            cache = SAECache()
        if self._b_dec is None:
            W = self.decoder.weight
            return torch.nn.functional.linear(x, W, None)
        return cache(self).decoder(x)
        # return self.decoder(x, cache=cache)
