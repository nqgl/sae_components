import torch
import torch.nn as nn
from torch import Tensor
from saeco.components.wrap import WrapsModule
from typing import Protocol, runtime_checkable, Optional
import saeco.core as cl
from abc import ABC


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


@runtime_checkable
class Resamplable(Protocol):
    def resample(self, *, indices, new_directions, bias_reset_value): ...


# @runtime_checkable
# class FeatureIndexable(Protocol):
#     @property
#     def feature_indexed(self) -> Tensor: ...


class ResampledWeight(WrapsModule):
    wrapped: HasFeatures

    def __init__(self, wrapped: HasFeatures):
        if not isinstance(wrapped, HasFeatures):
            raise TypeError(
                f"Expected HasFeatures, but {type(wrapped)} does not implement HasFeatures protocol."
            )
        # assert isinstance(wrapped, Resamplable)
        super().__init__(wrapped)

    def resample(self, *, indices, new_directions, bias_reset_value):
        self.wrapped.features[indices] = new_directions


class ResampledBias(WrapsModule):
    wrapped: HasFeatures

    def __init__(self, wrapped: HasFeatures):
        if not isinstance(wrapped, HasFeatures):
            raise TypeError("")
        # assert isinstance(wrapped, Resamplable)
        super().__init__(wrapped)

    def resample(self, *, indices, new_directions, bias_reset_value):
        self.wrapped.features[indices] = bias_reset_value


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
