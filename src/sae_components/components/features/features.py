import torch
import torch.nn as nn
from torch import Tensor
from sae_components.components.wrap import WrapsModule
from typing import Protocol, runtime_checkable, Optional
import sae_components.core as cl
from abc import ABC, abstractmethod


# class Features:
#     def __init__(self, features: nn.Parameter, transformation: callable):
#         self.features = features
#         self.transformation = transformation

#     def __getitem__(self, index):
#         return self.transformation(self.features)[index]

#     def __setitem__(self, index, value):
#         self.transformation(self.features)[index] = value

#     @property
#     def data(self):
#         return self.transformation(self.features)

#     @property
#     def grad(self):
#         return self.transformation(self.features.grad)


@runtime_checkable
class HasFeatures(Protocol):
    @property
    def features(self) -> Tensor: ...

    @property
    def features_grad(self) -> Optional[Tensor]: ...


# class FeaturesFromTransformMixin(ABC):
#     @abstractmethod
#     def transform(self, tensor: Tensor) -> Tensor: ...

#     @property
#     def features(self) -> Tensor:
#         return self.transform(self.wrapped.features)

#     @property
#     def features_grad(self) -> Tensor: ...


@runtime_checkable
class FeatureIndexable(Protocol):
    @property
    def feature_indexed(self) -> Tensor: ...


class ResampledWeight(WrapsModule):
    wrapped: HasFeatures

    def __init__(self, wrapped: HasFeatures):
        if not isinstance(wrapped, HasFeatures):
            raise TypeError(
                f"Expected HasFeatures, but {type(wrapped)} does not implement HasFeatures protocol."
            )
        # assert isinstance(wrapped, Resamplable)
        super().__init__(wrapped)

    def resample(self, *, indices, new_directions):
        self.wrapped.features[indices] = new_directions


class ResampledBias(WrapsModule):
    wrapped: HasFeatures

    def __init__(self, wrapped: FeatureIndexable, bias_reset_value=0):
        if not isinstance(wrapped, FeatureIndexable):
            raise TypeError("")
        # assert isinstance(wrapped, Resamplable)
        super().__init__(wrapped)
        self.bias_reset_value = bias_reset_value

    def resample(self, *, indices, new_directions):
        self.wrapped.feature_indexed[indices] = self.bias_reset_value


class MatMulWeights(WrapsModule):
    wrapped: cl.ops.MatMul

    def __init__(self, wrapped: cl.ops.MatMul):
        if isinstance(wrapped, nn.Parameter):
            wrapped = cl.ops.MatMul(wrapped)
        assert isinstance(wrapped, cl.ops.MatMul)
        super().__init__(wrapped)

    @abstractmethod
    def features_transform(self, tensor: Tensor) -> Tensor: ...

    @property
    def features(self) -> Tensor:
        return self.features_transform(self.wrapped.right.data)

    @property
    def features_grad(self) -> Optional[Tensor]:
        grad = self.wrapped.right.grad
        if grad is None:
            return None
        return self.features_transform(grad)

    def resampled(self):
        return ResampledWeight(self)


class DecoderWeights(MatMulWeights):
    def features_transform(self, tensor: Tensor) -> Tensor:
        return tensor


class EncoderWeights(MatMulWeights):
    wrapped: cl.ops.MatMul

    def features_transform(self, tensor: Tensor) -> Tensor:
        return tensor.transpose(-2, -1)


class LinWeights(WrapsModule):
    wrapped: nn.Linear

    def __init__(self, wrapped: nn.Linear):
        assert isinstance(wrapped, nn.Linear)
        super().__init__(wrapped)

    @abstractmethod
    def features_transform(self, tensor: Tensor) -> Tensor: ...

    @property
    def features(self) -> Tensor:
        f = self.features_transform(self.wrapped.weight.data)
        assert f.shape[0] != 768
        return f

    @property
    def features_grad(self) -> Optional[Tensor]:
        grad = self.wrapped.weight.grad
        if grad is None:
            return None
        f = self.features_transform(grad)
        assert f.shape[0] != 768
        return f

    def resampled(self):
        return ResampledWeight(self)  # TODO bias res too


class LinDecoder(LinWeights):
    def features_transform(self, tensor: Tensor) -> Tensor:
        return tensor.transpose(-2, -1)


class LinEncoder(LinWeights):
    def features_transform(self, tensor: Tensor) -> Tensor:
        return tensor


class EncoderBias(WrapsModule):
    wrapped: cl.ops.Add

    # TODO maybe make this wrap a param instead of the Add op
    def __init__(self, wrapped: cl.ops.Add):
        if isinstance(wrapped, nn.Parameter):
            wrapped = cl.ops.Add(wrapped)
        assert isinstance(wrapped, cl.ops.Add)
        super().__init__(wrapped)

    def resampled(self):
        return ResampledBias(self)

    @property
    def feature_indexed(self) -> Tensor:
        assert self.wrapped.bias.data.ndim == 1
        return self.wrapped.bias.data


class NormFeatures(WrapsModule):
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
        norm = torch.norm(features, dim=-1, keepdim=True)
        if (norm == 0).any():
            print("Norm is zero, not normalizing.")
            return
        self.wrapped.features[:] = features / norm


class Resamplable(Protocol):
    def resample(self, *, indices, new_directions): ...


class OrthogonalizeFeatureGrads(WrapsModule):
    wrapped: HasFeatures
    t: int

    def __init___(self, wrapped: HasFeatures):
        assert isinstance(
            wrapped, HasFeatures
        ), "Module must have features attribute (eg, make it Encoder/Decoder)"
        super().__init__(wrapped)

    def post_backward_hook(self):
        self.orthogonalize_features()

    @torch.no_grad()
    def orthogonalize_features(self):
        features = self.wrapped.features
        grad = self.wrapped.features_grad
        dec_normed = features / features.norm(dim=-1, keepdim=True)
        grad_orth = grad - (dec_normed * grad).sum(-1, keepdim=True) * dec_normed
        test = grad_orth * dec_normed + grad
        if grad.isinf().any():
            print("Infs in grads! ignoring.")
        if grad.isnan().any():
            print("NaNs in grads! returning")
            return
        if test.isinf().any():
            print("Infs in test! ignoring.")
        if test.isnan().any():
            print("NaNs in test! returning")
            return
        grad[:] = grad_orth
        assert (grad * dec_normed).sum(
            -1
        ).abs().mean() < 1e-1, f"Not orthogonal, oops. How not orthogonal? This much (max): {(grad * features).sum(-1).abs().max()}"


def check(t, i=""):
    if t.isnan().any():
        return f"NaNs in tensor{i}!"
    if t.isinf().any():
        return f"Infs in tensor{i}!"


def chl(*tl):
    for i, t in enumerate(tl):
        print(check(t, i))
