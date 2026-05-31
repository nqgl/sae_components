from functools import cached_property
from typing import Self

import torch
import torch.nn as nn
from torch import Tensor

from saeco.components.features.features_param import FeaturesView, FPInfo
from saeco.core.basic_ops import Sub


class DetachedLinear(nn.Module):
    """A linear view of ``lin`` with independently stop-gradient-able parts.

    ``detach_weight`` / ``detach_bias`` control whether the weight / bias flow
    gradients; ``use_bias=False`` drops the bias entirely. The parameters are
    always shared with ``lin`` (no copy) — only the autograd connection is cut
    where requested. Prefer constructing this via ``LinEncoder``/``LinDecoder``'s
    :meth:`LinWeightsMixin.as_detached`.
    """

    def __init__(self, lin, *, detach_weight=True, detach_bias=True, use_bias=True):
        super().__init__()
        self.lin = lin
        self.detach_weight = detach_weight
        self.detach_bias = detach_bias
        self.use_bias = use_bias

    def forward(self, x):
        weight = self.lin.weight.detach() if self.detach_weight else self.lin.weight
        bias = None
        if self.use_bias and self.lin.bias is not None:
            bias = self.lin.bias.detach() if self.detach_bias else self.lin.bias
        return torch.nn.functional.linear(x, weight, bias)


class LinWeightsMixin:
    self: nn.Linear
    weight: Tensor
    bias: nn.Parameter | None
    out_features: int

    @cached_property
    def features(self) -> FPInfo: ...

    def get_bias(self):
        return self.bias

    def set_resampled(self, resample=True) -> Self:
        for fp in self.features:
            fp.resampled = resample
        return self

    def resampled(self, resample: bool = True) -> Self:
        """Backwards-compatible alias for :meth:`set_resampled`; returns self."""
        return self.set_resampled(resample)

    @cached_property
    def sub_bias(self) -> Sub:
        """A ``Sub`` op that subtracts this layer's (shared) bias parameter."""
        assert self.bias is not None
        return Sub(self.bias)

    def as_detached(
        self,
        *,
        detach_weight: bool = True,
        detach_bias: bool = True,
        use_bias: bool = True,
    ) -> DetachedLinear:
        """A stop-gradient view of this layer (parameters shared, not copied).

        Defaults to fully detached. Common variants: ``as_detached()`` (detach
        all), ``as_detached(use_bias=False)`` (detach weight, drop bias),
        ``as_detached(detach_weight=False)`` (train the weight, detach the bias).
        """
        return DetachedLinear(
            self,
            detach_weight=detach_weight,
            detach_bias=detach_bias,
            use_bias=use_bias,
        )

    def new_bias(self) -> nn.Parameter:
        """A fresh zero bias parameter shaped like this layer's output."""
        return nn.Parameter(torch.zeros(self.out_features))


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
