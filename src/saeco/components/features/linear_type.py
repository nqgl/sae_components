from saeco.components.features.features import ResampledWeight
from saeco.components.wrap import WrapsModule
from typing import Optional

import torch.nn as nn
from torch import Tensor


from abc import abstractmethod
from saeco.components.features.features_param import FeaturesParam


# @property
# def features_grad(self) -> Optional[Tensor]: ...


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

    def resampled(self):
        return ResampledWeight(self)  # TODO bias res too

    def set_resampled(self, resample=True) -> "LinDecoder":
        for k, fp in self.features.items():
            fp.resampled = resample


from saeco.misc import lazycall, lazyprop


class LinDecoder(LinWeights):
    @property
    @lazycall
    def features(self) -> dict[str, FeaturesParam]:
        return {
            "weight": FeaturesParam(self.wrapped.weight, feature_index=1, fptype="dec")
        }

    # def features_transform(self, tensor: Tensor) -> Tensor:
    #     return tensor.transpose(-2, -1)


class LinEncoder(LinWeights):
    @property
    @lazycall
    def features(self) -> dict[str, FeaturesParam]:
        return {
            "weight": FeaturesParam(self.wrapped.weight, feature_index=0, fptype="enc"),
            "bias": FeaturesParam(self.wrapped.bias, feature_index=0, fptype="bias"),
        }

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
