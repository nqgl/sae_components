from saeco.components.features.features import ResampledWeight
from saeco.components.wrap import WrapsModule
from typing import Optional

import torch.nn as nn
from torch import Tensor

from functools import cached_property
from abc import abstractmethod
from saeco.components.features.features_param import FeaturesParam


class LinWeights(WrapsModule):
    wrapped: nn.Linear

    def __init__(self, wrapped: nn.Linear):
        assert isinstance(wrapped, nn.Linear)
        super().__init__(wrapped)

    # @abstractmethod
    # def features_transform(self, tensor: Tensor) -> Tensor: ...

    @property
    def features(self) -> dict[str, FeaturesParam]: ...

    # f = self.features_transform(self.wrapped.weight.data)

    # @property
    # def features_grad(self) -> Optional[Tensor]:
    #     grad = self.wrapped.weight.grad
    #     if grad is None:
    #         return None
    #     f = self.features_transform(grad)
    #     assert f.shape[0] != 768
    #     return f
    def get_weight(self):
        return self.wrapped.weight

    def get_bias(self):
        return self.wrapped.bias

    def resampled(self):
        return ResampledWeight(self)  # TODO bias res too

    def set_resampled(self, resample=True) -> "LinDecoder":
        for k, fp in self.features.items():
            fp.resampled = resample
        return self


class LinDecoder(LinWeights):
    @cached_property
    def features(self) -> dict[str, FeaturesParam]:
        return {
            "weight": FeaturesParam(self.get_weight(), feature_index=1, fptype="dec")
        }

    def __getitem__(self, key):
        lin = nn.Linear(self.wrapped.in_features, self.wrapped.out_features)
        lin.weight = self.features["weight"][key]
        lin.bias = self.wrapped.bias

    # def features_transform(self, tensor: Tensor) -> Tensor:
    #     return tensor.transpose(-2, -1)


class LinEncoder(LinWeights):
    @cached_property
    def features(self) -> dict[str, FeaturesParam]:
        d = {
            "weight": FeaturesParam(self.get_weight(), feature_index=0, fptype="enc"),
        }
        if self.bias is not None:
            d["bias"] = FeaturesParam(self.get_bias(), feature_index=0, fptype="bias")
        return d


class LongEncoder(LinWeights):  # TODO wip
    def __init__(self, wrapped: nn.Linear, split: list[int], bfs: list):
        super().__init__(wrapped)
        self.split = split
        self.bfs = bfs

    @cached_property
    def features(self) -> dict[str, FeaturesParam]:

        d = {
            "weight": FeaturesParam(self.get_weight(), feature_index=0, fptype="enc"),
        }
        weight = self.get_weight().split(self.split, dim=0)
        for i, (s, w) in enumerate(weight):
            d[f"weight_{i}"] = FeaturesParam(
                weight,
                feature_index=0,
                fptype="enc" if i == 0 else "other",
            )
        if self.bias is not None:
            bias = self.get_bias().split(self.split, dim=0)
            for i, (s, b) in enumerate(zip(self.split, bias)):
                d[f"bias{i}"] = FeaturesParam(b, feature_index=0, fptype="bias")
        return d

    # def features_transform(self, tensor: Tensor) -> Tensor:
    #     return tensor


# class ResampledLinEnc(ResampledWeight):
#     wrapped: LinEncoder

#     def __init__(self, wrapped: LinEncoder):
#         assert isinstance(wrapped, LinEncoder)
#         super().__init__(wrapped)

#     def resample(self, *, indices, new_directions, bias_reset_value):
#         super().resample(  # do weights
#             indices=indices,
#             new_directions=new_directions,
#             bias_reset_value=bias_reset_value,
#         )
#         if not self.bias is None:
#             self.bias.data[indices] = bias_reset_value


# class LinEncoder(LinWeights):
#     # def features_transform(self, tensor: Tensor) -> Tensor:
#     #     return tensor

#     def resampled(self):
#         return ResampledLinEnc(self)
