import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float
from typing import Protocol, runtime_checkable
import saeco.core as cl
import saeco.components as cc

# from saeco.components import (
#     Penalty,
#     L1Penalty,
#     FreqTracked,
#     EMAFreqTracker,
#     FreqTracker,
#     ResampledLayer,
#     Loss,
#     L2Loss,
#     SparsityPenalty,
#     SAECache,
#     NegBias,
# )


class Resampled: ...


def get_resample_sites(model: nn.Module):
    l = []
    for m in model.modules():
        if isinstance(m, cc.ResampledLayer):
            l.append(m)
    return l


@runtime_checkable
class FeatureResettable(Protocol):
    def reset_features(self, indices, directions): ...


class Resampler:
    def __init__(self, seq):
        self.seq = seq

    def get_feature_indices_to_reset(self): ...
    def get_reset_feature_directions(self): ...

    def resample(self, data_source):
        for m in self.seq:
            i = self.get_feature_indices_to_reset()
            d = self.get_reset_feature_directions()
            if isinstance(m, FeatureResettable):
                m.reset_features(indices=i, directions=d)


class ModularResamplerMaybe:
    # so this would like go around the enc and dec. it's got more crosstalk between hierarchies?
    # only matters for multi-layer, and even then i don't know if it's super important there

    def __init__(self, in_res, out_res):
        self.in_res = in_res
        self.out_res = out_res
