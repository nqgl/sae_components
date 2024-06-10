import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from abc import ABC, abstractmethod
from unpythonic import box
from typing import Protocol

import saeco.core as cl


class Normalizer(cl.Module, ABC):
    def normalize(self, x, *, cache: cl.Cache, **kwargs):
        x_normed, nfac = self._normalize(x)
        cache.normalization_factor = nfac
        return x_normed, nfac

    def _normalize(self, x):
        fac = self._get_normalization_factor(x) / (x.shape[-1] ** 0.5)
        return x / fac, fac

    @abstractmethod
    def _get_normalization_factor(self, x):
        raise NotImplementedError

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return self.normalize(x, cache=cache)

    def prime_normalizer(self, buffer, n=100):
        pass

    def io_normalize(self, module) -> "NormalizedIO":
        return NormalizedIO(model=module, normalizer=self)

    def input_normalize(self, module) -> "NormalizedInputs":
        return NormalizedInputs(model=module, normalizer=self)


class L2Normalizer(Normalizer):
    def _get_normalization_factor(self, x):
        return torch.linalg.vector_norm(x, dim=-1, ord=2, keepdim=True)


class ConstL2Normalizer(Normalizer):
    def __init__(self):
        super().__init__()
        self.register_buffer("est_avg_norm", torch.zeros(0))

    def prime_normalizer(self, buffer, n=100):
        norms = []
        for _ in range(n):
            sample = next(buffer)
            norms.append(torch.linalg.norm(sample, ord=2, dim=-1).mean())
        self.est_norm = torch.tensor(norms).mean()

    def _get_normalization_factor(self, x):
        return self.est_norm


class Normalized(cl.Seq):
    model: cl.Module
    normalizer: Normalizer


class NormalizedIO(Normalized):
    def __init__(self, model, normalizer):
        assert isinstance(normalizer, Normalizer)
        super().__init__(
            normalizer=cl.ReuseForward(normalizer),
            normalized=cl.Router(
                model=cl.ReuseForward(model), factor=cl.ops.Identity()
            ).reduce(lambda pred_normed, scale: pred_normed * scale),
        )


class NormalizedInputs(Normalized):
    def __init__(self, model, normalizer):
        assert isinstance(normalizer, Normalizer)
        super().__init__(
            normalizer=cl.ReuseForward(normalizer),
            normalized=cl.Router(
                model=cl.ReuseForward(model), factor=cl.ops.Identity()
            ).reduce(lambda *l: l[0]),
        )
