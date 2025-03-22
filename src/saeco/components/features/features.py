from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor
from saeco.components.type_acc_methods import (
    post_backward_hook,
    PostBackwardHook,
    post_step_hook,
)

import saeco.core as cl
from saeco.components.features.features_param import FeaturesParam, HasFeatures
from saeco.components.wrap import WrapsModule


@runtime_checkable  # TODO
class Resamplable(Protocol):
    def resample(self, *, indices, new_directions, bias_reset_value): ...


# @runtime_checkable
# class FeatureIndexable(Protocol):
#     @property
#     def feature_indexed(self) -> Tensor: ...


class ResampledWeight(WrapsModule):  # TODO dep
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


class ResampledBias(WrapsModule):  # TODO
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

    def resampled(self):  # todo
        raise NotImplementedError
        return ResampledBias(self)

    @property
    def features(self) -> dict[str, FeaturesParam]:
        return {
            "bias": FeaturesParam(
                self.wrapped.bias, feature_index=0, feature_parameter_type="bias"
            )
        }

    # @property
    # def feature_indexed(self) -> Tensor:
    #     assert self.wrapped.bias.data.ndim == 1
    #     return self.wrapped.bias.data


class NormFeatures(WrapsModule):
    wrapped: HasFeatures

    def __init__(self, wrapped: HasFeatures, index=None, ord=2, max_only=False):
        super().__init__(wrapped)
        self.index = index
        self.ord = ord
        self.max_only = max_only

    def post_step_hook(self):
        self.normalize_features()

    @torch.no_grad()
    def normalize_features(self):
        if self.index is None:
            fps = list(self.wrapped.features.values())
            assert len(fps) == 1
            fp = fps[0]
            assert fp.type == "dec"
        else:
            fp = self.wrapped.features[self.index]
        norm = torch.linalg.vector_norm(fp.features, dim=-1, keepdim=True, ord=self.ord)
        if (norm == 0).any():
            print("Norm is zero, not normalizing.")
            return
        if not self.max_only:
            fp.features[:] = fp.features / norm
        else:
            fp.features[:] = torch.where(norm > 1, fp.features / norm, fp.features)


class OrthogonalizeFeatureGrads(WrapsModule):
    wrapped: HasFeatures
    t: int

    def __init__(self, wrapped: HasFeatures, index=None):
        super().__init__(wrapped)
        self.index = index

    def post_backward_hook2(self):
        g = self.orthogonalize_feature_grads()
        b = g + 1

    @post_backward_hook
    @torch.no_grad()
    def orthogonalize_feature_grads(self) -> int:
        return 2
        if self.index is None:
            fps = list(self.wrapped.features.values())
            assert len(fps) == 1
            fp = fps[0]
            assert fp.type == "dec"
        else:
            fp = self.wrapped.features[self.index]
        if fp.grad is None:
            return
        dec_normed = fp.features / fp.features.norm(dim=-1, keepdim=True)
        grad_orth = fp.grad - (dec_normed * fp.grad).sum(-1, keepdim=True) * dec_normed
        test = grad_orth * dec_normed + fp.grad
        if fp.grad.isinf().any():
            print("Infs in grads! ignoring.")
        if fp.grad.isnan().any():
            print("NaNs in grads! returning")
            return
        if test.isinf().any():
            print("Infs in test! ignoring.")
        if test.isnan().any():
            print("NaNs in test! returning")
            return
        assert (
            grad_orth / (grad_orth.norm(dim=-1, keepdim=True) + 1e-6) * dec_normed
        ).sum(
            -1
        ).abs().mean() < 1e-4, f"Not orthogonal, oops. How not orthogonal? This much (max): {(fp.grad * fp.features).sum(-1).abs().max()}"
        fp.grad[:] = grad_orth


def check(t, i=""):
    if t.isnan().any():
        return f"NaNs in tensor{i}!"
    if t.isinf().any():
        return f"Infs in tensor{i}!"


def chl(*tl):
    for i, t in enumerate(tl):
        print(check(t, i))


def make_norm_features_mixin(index=None, ord=2, max_only=False):

    class NormFeaturesMixin:
        features: dict[str, FeaturesParam]

        @post_step_hook
        @torch.no_grad()
        def normalize_features(self):
            if index is None:
                fps = list(self.features.values())
                assert len(fps) == 1
                fp = fps[0]
                assert fp.type == "dec"
            else:
                fp = self.features[index]
            norm = torch.linalg.vector_norm(fp.features, dim=-1, keepdim=True, ord=ord)
            if (norm == 0).any():
                print("Norm is zero, not normalizing.")
                return
            if not max_only:
                fp.features[:] = fp.features / norm
            else:
                fp.features[:] = torch.where(norm > 1, fp.features / norm, fp.features)

    return NormFeaturesMixin


NormFeaturesMixin = make_norm_features_mixin()


def make_orthogonalize_feature_grads_mixin(index=None):
    class OrthogonalizeFeatureGradsMixin:
        features: dict[str, FeaturesParam]
        t: int

        # could be similar to an arch_prop in mechanism
        @post_backward_hook
        @torch.no_grad()  # want the post backward hook decorator thing here
        def orthogonalize_feature_grads(
            self,
        ):
            if index is None:
                fps = list(self.features.values())
                assert len(fps) == 1
                fp = fps[0]
                assert fp.type == "dec"
            else:
                fp = self.features[index]
            if fp.grad is None:
                return
            dec_normed = fp.features / fp.features.norm(dim=-1, keepdim=True)
            grad_orth = (
                fp.grad - (dec_normed * fp.grad).sum(-1, keepdim=True) * dec_normed
            )
            test = grad_orth * dec_normed + fp.grad
            if fp.grad.isinf().any():
                print("Infs in grads! ignoring.")
            if fp.grad.isnan().any():
                print("NaNs in grads! returning")
                return
            if test.isinf().any():
                print("Infs in test! ignoring.")
            if test.isnan().any():
                print("NaNs in test! returning")
                return
            assert (
                grad_orth / (grad_orth.norm(dim=-1, keepdim=True) + 1e-6) * dec_normed
            ).sum(
                -1
            ).abs().mean() < 1e-4, f"Not orthogonal, oops. How not orthogonal? This much (max): {(fp.grad * fp.features).sum(-1).abs().max()}"
            fp.grad[:] = grad_orth
            return 1

    return OrthogonalizeFeatureGradsMixin


OrthogonalizeFeatureGradsMixin = make_orthogonalize_feature_grads_mixin()
