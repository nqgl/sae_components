import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from abc import ABC, abstractmethod
from unpythonic import box


class NormalizerMixinBase(nn.Module, ABC):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.d_data = cfg.d_data

    def normalize(self, x, invert_normalization_box: box = None):
        if invert_normalization_box is None:
            invert_normalization_box = box()
        return self._normalize(x, invert_normalization_box=invert_normalization_box)

    def _normalize(self, x, invert_normalization_box):
        fac = self._get_normalization_factor(x) / self.d_data**0.5
        invert_normalization_box << fac
        return x / fac

    def _get_normalization_factor(self, x):
        raise NotImplementedError

    def forward(self, x, **kwargs):
        invert_normalization_box = box()
        x_normed = self._normalize(x, invert_normalization_box=invert_normalization_box)
        return super().forward(x_normed, **kwargs) * invert_normalization_box.get()

    def prime_normalizer(self, buffer, n=10):
        pass


class L2NormalizerMixin(NormalizerMixinBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)

    def _get_normalization_factor(self, x):
        return torch.linalg.vector_norm(x, dim=-1, ord=2, keepdim=True)


class ConstL2NormalizerMixin(NormalizerMixinBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.register_buffer("est_avg_norm", torch.zeros(0))

    def prime_normalizer(self, buffer, n=10):
        norms = []
        for _ in range(n):
            sample = next(buffer)
            norms.append(torch.linalg.norm(sample, ord=2, dim=-1).mean())
        self.est_norm = torch.tensor(norms).mean()

    def _get_normalization_factor(self, x):
        return self.est_norm
