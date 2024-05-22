import torch
import torch.nn as nn
from torch import Tensor
from sae_components.components.decoder_normalization.wrap import WrapMix
from typing import Protocol, runtime_checkable
import sae_components.core as cl
from abc import ABC, abstractmethod

# @runtime_checkable
# class FeatureIndexable(Protocol):
#     @property
#     def feature_indexed(self) -> Tensor: ...


@runtime_checkable
class HasFeatures(Protocol):
    @property
    def features(self) -> Tensor: ...


class MatMulWeights(WrapMix, ABC):

    wrapped: cl.ops.MatMul

    def __init__(self, wrapped: cl.ops.MatMul):
        if isinstance(wrapped, nn.Parameter):
            wrapped = cl.ops.MatMul(wrapped)
        assert isinstance(wrapped, cl.ops.MatMul)
        super().__init__(wrapped)

    @property
    @abstractmethod
    def features(self) -> Tensor: ...


class DecoderWeights(MatMulWeights):
    @property
    def features(self) -> Tensor:
        return self.wrapped.weight.data

    @features.setter
    def features(self, value: Tensor):
        raise NotImplementedError(
            "This is a read-only property -- if you need to edit features, do an in place operation on the features tensor."
        )


class EncoderWeights(MatMulWeights):
    wrapped: cl.ops.MatMul

    @property
    def features(self) -> Tensor:
        return self.wrapped.weight.data.transpose(-2, -1)

    @features.setter
    def features(self, value: Tensor):
        raise NotImplementedError(
            "This is a read-only property -- if you need to edit features, do an in place operation on the features tensor."
        )


class EncoderBias(WrapMix):
    wrapped: cl.ops.Add

    def __init__(self, wrapped: cl.ops.Add):
        if isinstance(wrapped, nn.Parameter):
            wrapped = cl.ops.Add(wrapped)
        assert isinstance(wrapped, cl.ops.Add)
        super().__init__(wrapped)


class NormFeatures(WrapMix):
    wrapped: HasFeatures

    def __init___(self, wrapped: HasFeatures):
        assert isinstance(
            wrapped, HasFeatures
        ), "Module must have features attribute (eg, make it Encoder/Decoder)"
        super().__init__(wrapped)

    def post_step_hook(self):
        self.normalize_features()

    @torch.no_grad()
    def normalize_features(self):
        features = self.wrapped.features
        self.wrapped.features[:] = features / torch.norm(features, dim=-1, keepdim=True)


class Resamplable(Protocol):
    def resample(self, *, indices, new_directions): ...


class ResampledWeight(WrapMix):
    wrapped: HasFeatures

    def __init__(self, wrapped: HasFeatures):
        if not isinstance(wrapped, HasFeatures):
            raise TypeError("")
        # assert isinstance(wrapped, Resamplable)
        super().__init__(wrapped)

    def resample(self, *, indices, new_directions):
        self.wrapped.features[indices] = new_directions


class ResampledBias(WrapMix):
    wrapped: HasFeatures

    def __init__(self, wrapped: HasFeatures, bias_value=0):
        if not isinstance(wrapped, HasFeatures):
            raise TypeError("")
        # assert isinstance(wrapped, Resamplable)
        super().__init__(wrapped)
        self.bias_value = bias_value

    def resample(self, *, indices, new_directions):
        self.wrapped.features[indices] = self.bias_value
