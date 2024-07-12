import torch
import torch.nn as nn
from torch import Tensor
from saeco.components.features.optim_reset import FeaturesParam
from saeco.components.wrap import WrapsModule
from typing import Protocol, runtime_checkable, Optional
import saeco.core as cl


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


# @runtime_checkable
# class HasFeatures(Protocol):
#     @property
#     def features(self) -> Tensor: ...

#     @property
#     def features_grad(self) -> Optional[Tensor]: ...


@runtime_checkable
class HasFeatures(Protocol):
    @property
    def features(self) -> dict[str, FeaturesParam]: ...


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
            "bias": FeaturesParam(self.wrapped.bias, feature_index=0, fptype="bias")
        }

    # @property
    # def feature_indexed(self) -> Tensor:
    #     assert self.wrapped.bias.data.ndim == 1
    #     return self.wrapped.bias.data


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
        fps = list(self.wrapped.features.values())
        assert len(fps) == 1
        fp = fps[0]
        assert fp.type == "dec"
        features = fp.features
        norm = torch.norm(fp.features, dim=-1, keepdim=True)
        if (norm == 0).any():
            print("Norm is zero, not normalizing.")
            return
        fp.features[:] = fp.features / norm


class OrthogonalizeFeatureGrads(WrapsModule):
    wrapped: HasFeatures
    t: int

    def __init___(self, wrapped: HasFeatures):
        assert isinstance(
            wrapped, HasFeatures
        ), "Module must have features attribute (eg, make it Encoder/Decoder)"
        super().__init__(wrapped)

    def post_backward_hook(self):
        self.orthogonalize_feature_grads()

    @torch.no_grad()
    def orthogonalize_feature_grads(self):
        fps = list(self.wrapped.features.values())
        assert len(fps) == 1
        fp = fps[0]
        assert fp.type == "dec"
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
        fp.grad[:] = grad_orth
        assert (fp.grad * dec_normed).sum(
            -1
        ).abs().mean() < 1e-1, f"Not orthogonal, oops. How not orthogonal? This much (max): {(fp.grad * fp.features).sum(-1).abs().max()}"


def check(t, i=""):
    if t.isnan().any():
        return f"NaNs in tensor{i}!"
    if t.isinf().any():
        return f"Infs in tensor{i}!"


def chl(*tl):
    for i, t in enumerate(tl):
        print(check(t, i))
