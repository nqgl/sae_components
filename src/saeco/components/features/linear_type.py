from functools import cached_property
from typing import Self

import torch.nn as nn
from torch import Tensor

from saeco.components.features.features_param import FeaturesParam


class LinWeightsMixin:
    self: nn.Linear
    weight: Tensor
    bias: nn.Parameter | None

    @cached_property
    def features(self) -> dict[str, FeaturesParam]: ...

    def get_bias(self):
        return self.bias

    def set_resampled(self, resample=True) -> Self:
        for _k, fp in self.features.items():
            fp.resampled = resample
        return self


class LinDecoderMixin(LinWeightsMixin):
    @cached_property
    def features(self) -> dict[str, FeaturesParam]:
        # bias is excluded for decoder because it cannot be indexed by feature
        return {
            "weight": FeaturesParam(
                self.weight,
                feature_index=1,
                feature_parameter_type=FeaturesParam.FPTYPES.dec,
            )
        }


class LinEncoderMixin(LinWeightsMixin):
    @cached_property
    def features(self) -> dict[str, FeaturesParam]:
        d = {
            "weight": FeaturesParam(
                self.weight,
                feature_index=0,
                feature_parameter_type=FeaturesParam.FPTYPES.enc,
            ),
        }
        if self.bias is not None:
            d["bias"] = FeaturesParam(
                self.bias,
                feature_index=0,
                feature_parameter_type=FeaturesParam.FPTYPES.bias,
            )
        return d


class LinEncoder(LinEncoderMixin, nn.Linear):
    pass


class LinDecoder(LinDecoderMixin, nn.Linear):
    pass
