from abc import ABC, abstractmethod
from typing import Protocol

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from unpythonic import box

import saeco.core as cl


class Normalizer(cl.Module, ABC):
    primed: bool

    def __init__(self, init):
        super().__init__()

    @abstractmethod
    def invert(self, x, *, cache: cl.Cache, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x, *, cache: cl.Cache, **kwargs):
        raise NotImplementedError

    def prime_normalizer(self, buffer, n=100):
        pass

    def io_normalize(self, module) -> "NormalizedIO":
        return NormalizedIO(model=module, normalizer=self)

    def input_normalize(self, module) -> "NormalizedInputs":
        return NormalizedInputs(model=module, normalizer=self)

    def output_denormalize(self, module) -> "DeNormalizedOutputs":
        return DeNormalizedOutputs(model=module, normalizer=self)

    def get_denormalizer(self):
        return DeNormalizer(self)


class DeNormalizer(cl.Module):
    def __init__(self, normalizer: Normalizer):
        super().__init__()
        self.normalizer = normalizer

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return self.normalizer.invert(x, cache=cache, **kwargs)


class Normalized(cl.Module):
    model: cl.Module
    normalizer: Normalizer

    def __init__(self, model: cl.Module, normalizer: Normalizer):
        super().__init__()
        self.model = cl.ReuseForward(model)
        self.normalizer = cl.ReuseForward(normalizer)
        self._normalizer = normalizer


class NormalizedIO(Normalized):
    def forward(self, x, *, cache: cl.Cache, **kwargs):
        x_normed = self.normalizer(x, cache=cache["normalization"])
        return self._normalizer.invert(
            self.model(x_normed, cache=cache["normalized"]),
            cache=cache["normalization"],
        )


# class NormalizedIO2(Normalized):
#     def __init__(
#         self,
#         model: cl.Module,
#         normalizer: Normalizer,
#         normalize_in=True,
#         denormalize_out=True,
#     ):
#         super().__init__(model, normalizer)
#         self.normalize_in = normalize_in
#         self.denormalize_out = denormalize_out


#     def forward(self, x, *, cache: cl.Cache, **kwargs):
#         if self.normalize_in:
#             x = self.normalizer(x, cache=cache["normalization"])
#         if self.denormalize_out:
#             return self._normalizer.invert(
#                 self.model(x, cache=cache["normalized"]),
#                 cache=cache["normalization"],
#             )
#         else:
#             return self.model(x, cache=cache["normalized"])


class NormalizedInputs(Normalized):
    def forward(self, x, *, cache: cl.Cache, **kwargs):
        x_normed = self.normalizer(x, cache=cache["normalization"])
        return self.model(x_normed, cache=cache["normalized"])


class DeNormalizedOutputs(Normalized):
    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return self._normalizer.invert(
            self.model(x, cache=cache["normalized"]), cache=cache["normalization"]
        )


class AffineNormalizer(Normalizer):
    def shift(self, x, *, cache) -> Tensor:
        raise NotImplementedError

    def scale(self, x, *, cache) -> Tensor:
        raise NotImplementedError

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        shift = self.shift(x, cache=cache)
        x_s = x - shift
        scale = self.scale(x_s, cache=cache)
        cache.shift = ...  # force watching for now
        cache.scale = ...  # may integrate cacheproc-like behavior later
        cache.shift = shift
        cache.scale = scale
        return (x_s) / scale

    def invert(self, x, *, cache: cl.Cache, **kwargs):
        return x * cache.scale + cache.shift


class LNNormalizer(AffineNormalizer):
    def __init__(self, init, eps=1e-07):
        super().__init__(init)
        self.eps = eps

    def shift(self, x, *, cache) -> Tensor:
        return x.mean(dim=-1, keepdim=True)

    def scale(self, x: Tensor, *, cache) -> Tensor:
        return x.std(dim=-1, keepdim=True, unbiased=False) + self.eps


class L2Normalizer(AffineNormalizer):
    def __init__(self, init, eps=1e-07):
        super().__init__(init)
        self.eps = eps

    def shift(self, x, *, cache) -> Tensor:
        return 0

    def scale(self, x: Tensor, *, cache) -> Tensor:
        return x.std(dim=-1, keepdim=True, unbiased=False) + self.eps


class ConstL2Normalizer(AffineNormalizer):
    def __init__(self, init):
        super().__init__(init)
        self.register_buffer("est_avg_norm", torch.zeros(0))

    def prime_normalizer(self, buffer, n=100):
        norms = []
        for _ in range(n):
            sample = next(buffer)
            norms.append(torch.linalg.norm(sample, ord=2, dim=-1).mean())
        self.est_norm = torch.tensor(norms).mean()

    def _get_normalization_factor(self, x):
        return self.est_norm


class ConstLNNormalizer(AffineNormalizer):
    def __init__(self, init):
        super().__init__(init)
        self.register_buffer("mean", torch.zeros(init.d_data))
        self.register_buffer("est_avg_norm", torch.zeros(0))

    @torch.no_grad()
    def prime_normalizer(self, buffer, n=100):
        means = []
        samples = []
        for _ in range(n):
            sample = next(buffer)
            samples.append(sample)
            means.append(sample.mean(dim=0))
        self.mean = torch.stack(means).mean(dim=0)
        norms = []
        for sample in samples:
            norms.append((sample - self.mean).std(dim=-1).mean())
        self.est_norm = torch.tensor(norms).mean()

    def scale(self, x, *, cache) -> Tensor:
        return self.est_norm

    def shift(self, x, *, cache) -> Tensor:
        return self.mean


class ElementwiseZdistNormalizer(AffineNormalizer):
    def __init__(self, init):
        super().__init__()
        self.register_buffer("std", torch.zeros(init.d_data))
        self.register_buffer("mean", torch.zeros(init.d_data))

    def prime_normalizer(self, buffer, n=100):
        means = []
        samples = []
        for _ in range(n):
            sample = next(buffer)
            samples.append(sample)
            means.append(sample.mean(dim=0))
        self.mean = torch.stack(means).mean(dim=0)
        norms = []
        for sample in samples:
            norms.append((sample - self.mean).std(dim=-1).mean())
        self.est_norm = torch.tensor(norms).mean()

    def _get_normalization_factor(self, x):
        return self.est_norm


class BatchNormalizer(AffineNormalizer):
    def __init__(self, init, eps=1e-07):
        super().__init__(init)
        self.eps = eps

    def shift(self, x, *, cache) -> Tensor:
        return x.mean(dim=0, keepdim=True)

    def scale(self, x: Tensor, *, cache) -> Tensor:
        return x.std(dim=0, keepdim=True, unbiased=False) + self.eps


from enum import IntEnum

# class Aggregation(SweepableConfig):
#     primed: bool
#     running: bool
#     batch: bool


from jaxtyping import Float

from saeco.sweeps import SweepableConfig


class GNConfig(SweepableConfig):
    class Aggregation(IntEnum):
        DONTUSE = 0
        PRIMED = 1
        RUNNING_AVG = 2
        BATCH_AVG = 3
        LEARNED = 4

    class SAggregation(IntEnum):
        DONTUSE = 0
        PRIMED = 1
        SAMPLE = 5
        # BATCH_AVG = 3
        # RUNNING_AVG = 4

    mu_s: SAggregation = SAggregation.PRIMED
    mu_e: Aggregation = Aggregation.DONTUSE
    std_s: SAggregation = SAggregation.PRIMED
    std_e: Aggregation = Aggregation.PRIMED
    sandwich: bool = False


Aggregation = GNConfig.Aggregation
SAggregation = GNConfig.SAggregation


class GeneralizedNormalizer(Normalizer):
    def __init__(self, init, cfg: GNConfig, eps=1e-07):
        super().__init__(init)
        self.eps = eps
        self.cfg = cfg
        self.primed = False
        if cfg.mu_e in (Aggregation.PRIMED, Aggregation.RUNNING_AVG):
            self.register_buffer(
                "_mu_e",
                torch.full(
                    (
                        1,
                        init.d_data,
                    ),
                    torch.nan,
                ),
            )
        elif cfg.mu_e == Aggregation.LEARNED:
            self._mu_e = nn.Parameter(
                torch.full(
                    (
                        1,
                        init.d_data,
                    ),
                    torch.nan,
                )
            )
        if cfg.std_e in (Aggregation.PRIMED, Aggregation.RUNNING_AVG):
            self.register_buffer(
                "_std_e",
                torch.full(
                    (
                        1,
                        init.d_data,
                    ),
                    torch.nan,
                ),
            )
        elif cfg.std_e == Aggregation.LEARNED:
            self._std_e = nn.Parameter(
                torch.full(
                    (
                        1,
                        init.d_data,
                    ),
                    torch.nan,
                )
            )
        self.cfg.sandwich = False
        if cfg.mu_s == SAggregation.PRIMED:
            self.register_buffer(
                "_mu_s",
                torch.full(
                    (),
                    torch.nan,
                ),
            )
        if cfg.std_s == SAggregation.PRIMED:
            self.register_buffer(
                "_std_s",
                torch.full(
                    (),
                    torch.nan,
                ),
            )

        assert not (
            self.cfg.sandwich
            and (cfg.mu_s == SAggregation.PRIMED or cfg.std_s == SAggregation.PRIMED)
        )

    @torch.no_grad()
    def prime_normalizer(self, buffer, n=20):
        assert not self.primed
        self.primed = True
        samples = [next(buffer) for _ in range(n)]
        x = torch.cat(samples, dim=0)
        # samples = [x - self.mu_s(x) for x in samples]

        if self.cfg.sandwich:
            x = x - self.mu_s(x)

        if self.cfg.mu_e not in (Aggregation.DONTUSE, Aggregation.BATCH_AVG):
            # mu_es = [self.elementwise_mean(x) for x in samples]
            self._mu_e.data = self.elementwise_mean(x)

        x = x - self.mu_e(x)

        if self.cfg.mu_s == SAggregation.PRIMED:
            self._mu_s.data = self.sample_mean(x).mean()

        x = x - self.mu_s(x)
        # samples = [x - self.mu_e(x) for x in samples]
        # samples = [x - self.mu_s(x) for x in samples]
        if self.cfg.sandwich:
            x = x / self.std_s(x)

        if self.cfg.std_e not in (Aggregation.DONTUSE, Aggregation.BATCH_AVG):
            # std_es = [self.elementwise_std(x) for x in samples]
            self._std_e.data = self.elementwise_std(x)
            # x = x / self.std_e(x)
            # x = x / self.std_s(x)

        x = x / self.std_e(x)
        if self.cfg.std_s == SAggregation.PRIMED:
            self._std_s.data = self.sample_std(x).mean()

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        # cache.mu_e = ...
        # cache.mu_e = mu_e
        # mu_e = self.mu_e(x, cache=cache)
        # x = x - mu_e

        # cache.mu_s = ...
        # cache.mu_s = mu_s
        # mu_s = self.mu_s(x, cache=cache)
        # x = x - mu_s

        # cache.std_e = ...
        # cache.std_e = std_e
        # std_e = self.std_e(x, cache=cache)
        # x = x / std_e

        # cache.std_s = ...
        # cache.std_s = std_s
        # std_s = self.std_s(x, cache=cache)
        # x = x / std_s
        ####
        if not self.cfg.sandwich:
            x = x - cache(self, force_watch=True).mu_e(x)
            x = x - cache(self, force_watch=True).mu_s(x)
            x = x / cache(self, force_watch=True).std_e(x)
            x = x / cache(self, force_watch=True).std_s(x)
            return x
        x0 = x

        mu_s_1 = self.mu_s(x)
        x = x - mu_s_1

        x = x - cache(self, force_watch=True).mu_e(x)

        mu_s_2 = self.mu_s(x)
        x = x - mu_s_2

        cache.mu_s = ...
        cache.mu_s = mu_s_1 + mu_s_2

        std_s_1 = self.std_s(x)
        x = x / std_s_1

        x = x / cache(self, force_watch=True).std_e(x)

        std_s_2 = self.std_s(x)
        x = x / std_s_2

        cache.std_s = ...
        cache.std_s = std_s_1 * std_s_2

        return (x0 - (cache.mu_s + cache.mu_e)) / (cache.std_s * cache.std_e)

    def sample_mean(self, x):
        return x.detach().float().mean(dim=-1, keepdim=True)

    def elementwise_mean(self, x):
        return x.detach().float().mean(dim=0, keepdim=True)

    def sample_std(self, x):
        return x.detach().float().std(dim=-1, keepdim=True, unbiased=False) + self.eps

    def elementwise_std(self, x):
        return x.detach().float().std(dim=0, keepdim=True, unbiased=False) + self.eps

    def invert(self, x, *, cache: cl.Cache, **kwargs):
        return x * (cache.std_s * cache.std_e) + (cache.mu_s + cache.mu_e)

    def mu_s(self, x, *, cache=None) -> Float[Tensor, "batch 1"]:
        if self.cfg.mu_s == SAggregation.DONTUSE:
            return 0
        elif self.cfg.mu_s == SAggregation.PRIMED:
            return self._mu_s
        elif self.cfg.mu_s == SAggregation.SAMPLE:
            return self.sample_mean(x)

    def std_s(self, x, *, cache=None) -> Float[Tensor, "batch 1"]:
        if self.cfg.std_s == SAggregation.DONTUSE:
            return 1
        elif self.cfg.std_s == SAggregation.PRIMED:
            return self._std_s
        elif self.cfg.std_s == SAggregation.SAMPLE:
            return self.sample_std(x)

    def mu_e(self, x, *, cache=None) -> Float[Tensor, "1 d_data"]:
        if self.cfg.mu_e == Aggregation.DONTUSE:
            return 0
        if self.cfg.mu_e == Aggregation.RUNNING_AVG:
            self._mu_e.data.lerp_(self.elementwise_mean(x), 0.003)
        if self.cfg.mu_e == Aggregation.BATCH_AVG:
            return self.elementwise_mean(x)
        return self._mu_e

    def std_e(self, x, *, cache=None) -> Float[Tensor, "1 d_data"]:
        if self.cfg.std_e == Aggregation.DONTUSE:
            return 1
        if self.cfg.std_e == Aggregation.RUNNING_AVG:
            self._std_e.data.lerp_(self.elementwise_std(x), 0.003)
        if self.cfg.std_e == Aggregation.BATCH_AVG:
            return self.elementwise_std(x)
        return torch.abs(self._std_e) + 1e-07


class StaticInvertibleGeneralizedNormalizer(GeneralizedNormalizer):
    def __init__(self, init, cfg: GNConfig, eps=1e-7):

        static_aggs = (
            Aggregation.DONTUSE,
            Aggregation.PRIMED,
            Aggregation.RUNNING_AVG,
            Aggregation.LEARNED,
        )

        static_saggs = (
            SAggregation.DONTUSE,
            SAggregation.PRIMED,
        )
        assert (
            cfg.mu_e in static_aggs
        ), f"{cfg.mu_e} is not a static aggregation but is being used with a static-invertible normalizer"
        assert (
            cfg.std_e in static_aggs
        ), f"{cfg.std_e} is not a static aggregation but is being used with a static-invertible normalizer"
        assert (
            cfg.mu_s in static_saggs
        ), f"{cfg.mu_s} is not a static aggregation but is being used with a static-invertible normalizer"
        assert (
            cfg.std_s in static_saggs
        ), f"{cfg.std_s} is not a static aggregation but is being used with a static-invertible normalizer"

        super().__init__(init, cfg, eps)

    def invert(self, x, *, cache: cl.Cache, **kwargs):
        return x * (self.std_s(None) * self.std_e(None)) + (
            self.mu_s(None) + self.mu_e(None)
        )


def main():
    class C: ...

    init = C()
    setattr(init, "d_data", 768)

    cfg = GNConfig(
        mu_s=True, mu_e=Aggregation.BATCH_AVG, std_s=True, std_e=Aggregation.BATCH_AVG
    )
    normalizer = GeneralizedNormalizer(init, cfg)
    cache = cl.Cache()
    x = torch.randn(10, 768, dtype=torch.float16) * 100
    x = x.cuda()
    with torch.autocast(device_type="cuda"):
        xn = normalizer(x, cache=cache)
        xdn = normalizer.invert(xn, cache=cache)
        print(x - xdn)


if __name__ == "__main__":
    main()
