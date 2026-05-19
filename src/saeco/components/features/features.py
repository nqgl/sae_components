from typing import Protocol, runtime_checkable
from warnings import deprecated

import torch.nn as nn

import saeco.core as cl
from saeco.components.features.features_param import FeaturesParam, HasFeatures
from saeco.components.wrap import WrapsModule


@runtime_checkable
class Resamplable(Protocol):
    def resample(self, *, indices, new_directions, bias_reset_value, optim): ...


@deprecated("resampled x usage")
class ResampledWeight(WrapsModule):  # TODO dep
    wrapped: HasFeatures

    def __init__(self, wrapped: HasFeatures):
        if not isinstance(wrapped, HasFeatures):
            raise TypeError(
                f"Expected HasFeatures, but {type(wrapped)} does not implement "
                "HasFeatures protocol."
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
