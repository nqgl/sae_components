from typing import Optional, Protocol, runtime_checkable
from typing_extensions import Self
import torch
import torch.nn as nn
from torch import Tensor
from saeco.components.type_acc_methods import (
    post_backward_hook,
    PostBackwardHook,
    post_step_hook,
)
from typing_extensions import deprecated
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


@deprecated("resampled x usage")
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


@deprecated("resampled x usage")
class ResampledBias(WrapsModule):  # TODO
    wrapped: HasFeatures

    def __init__(self, wrapped: HasFeatures):
        if not isinstance(wrapped, HasFeatures):
            raise TypeError("")
        # assert isinstance(wrapped, Resamplable)
        super().__init__(wrapped)

    def resample(self, *, indices, new_directions, bias_reset_value):
        self.wrapped.features[indices] = bias_reset_value


@deprecated("EncoderBias as wrapper will be replaced by an Add subclass")
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
                self.wrapped.bias,
                feature_index=0,
                feature_parameter_type=FeaturesParam.FPTYPES.bias,
            )
        }

    # @property
    # def feature_indexed(self) -> Tensor:
    #     assert self.wrapped.bias.data.ndim == 1
    #     return self.wrapped.bias.data


@deprecated("transitioning to mixins over wrappers")
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


@deprecated("transitioning to mixins over wrappers")
class OrthogonalizeFeatureGrads(WrapsModule):
    wrapped: HasFeatures
    t: int

    def __init__(self, wrapped: HasFeatures, index=None):
        super().__init__(wrapped)
        self.index = index

    def post_backward_hook(self):
        g = self.orthogonalize_feature_grads()

    @torch.no_grad()
    def orthogonalize_feature_grads(self):
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


# def check(t, i=""):
#     if t.isnan().any():
#         return f"NaNs in tensor{i}!"
#     if t.isinf().any():
#         return f"Infs in tensor{i}!"


# def chl(*tl):
#     for i, t in enumerate(tl):
#         print(check(t, i))


class NormFeaturesMixin:
    features: dict[str, FeaturesParam]
    _norm_features_mixin_featureparameter_name: str | None = None
    _norm_features_mixin_ord = 2
    _norm_features_mixin_max_only = False

    @post_step_hook
    @torch.no_grad()
    def normalize_features(self):
        if self._norm_features_mixin_featureparameter_name is None:
            fps = list(self.features.values())
            assert len(fps) == 1
            fp = fps[0]
            assert fp.type == "dec"
        else:
            fp = self.features[self._norm_features_mixin_featureparameter_name]
        norm = torch.linalg.vector_norm(
            fp.features, dim=-1, keepdim=True, ord=self._norm_features_mixin_ord
        )
        if (norm == 0).any():
            print("Norm is zero, not normalizing.")
            return
        if not self._norm_features_mixin_max_only:
            fp.features[:] = fp.features / norm
        else:
            fp.features[:] = torch.where(norm > 1, fp.features / norm, fp.features)

    @classmethod
    def mixin_with_parameters(
        cls,
        featureparameter_name: str | None = None,
        ord: float = 2,
        only_clip_norms: bool = False,
    ) -> type[Self]:
        assert cls is NormFeaturesMixin

        class ParameterizedNormFeaturesMixin(cls):
            _norm_features_mixin_featureparameter_name = featureparameter_name
            _norm_features_mixin_ord = ord
            _norm_features_mixin_max_only = only_clip_norms

        return ParameterizedNormFeaturesMixin


class OrthogonalizeFeatureGradsMixin:
    features: dict[str, FeaturesParam]
    t: int
    _orthogonalize_feature_grads_mixin_featureparameter_name: str | None = None

    @post_backward_hook
    @torch.no_grad()
    def orthogonalize_feature_grads(
        self,
    ):
        if self._orthogonalize_feature_grads_mixin_featureparameter_name is None:
            fps = list(self.features.values())
            assert len(fps) == 1
            fp = fps[0]
            assert fp.type == "dec"
        else:
            fp = self.features[
                self._orthogonalize_feature_grads_mixin_featureparameter_name
            ]
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
        return 1

    @classmethod
    def mixin_with_parameters(
        cls,
        featureparameter_name: str | None = None,
    ) -> type[Self]:
        assert cls is OrthogonalizeFeatureGradsMixin

        class ParameterizedOrthogonalizeFeatureGradsMixin(cls):
            _orthogonalize_feature_grads_mixin_featureparameter_name = (
                featureparameter_name
            )

        return ParameterizedOrthogonalizeFeatureGradsMixin
