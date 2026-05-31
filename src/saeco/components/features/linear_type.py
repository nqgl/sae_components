from functools import cached_property
from typing import Self

import torch.nn as nn
from torch import Tensor

from saeco.components.features.features_param import FeaturesView, FPInfo


class LinWeightsMixin:
    self: nn.Linear
    weight: Tensor
    bias: nn.Parameter | None

    @cached_property
    def features(self) -> FPInfo: ...

    def get_bias(self):
        return self.bias

    def set_resampled(self, resample=True) -> Self:
        for fp in self.features:
            fp.resampled = resample
        return self


class LinDecoderMixin(LinWeightsMixin):
    @cached_property
    def features(self) -> FPInfo:
        # bias is excluded for decoder because it cannot be indexed by feature
        return FPInfo(dec=FeaturesView(self.weight, feature_index=1))


class LinEncoderMixin(LinWeightsMixin):
    @cached_property
    def features(self) -> FPInfo:
        return FPInfo(
            enc=FeaturesView(self.weight, feature_index=0),
            bias=(
                None if self.bias is None else FeaturesView(self.bias, feature_index=0)
            ),
        )


class LinEncoder(LinEncoderMixin, nn.Linear):
    pass


class LinDecoder(LinDecoderMixin, nn.Linear):
    pass
