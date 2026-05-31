from typing import Any, Protocol, Self, runtime_checkable

from .features_param import FeatureParamType, FeaturesView


class ParamMetadata:
    def __init__(self, param, param_type=None):
        self.param = param
        self.as_features: FeaturesView | None = None
        assert not hasattr(param, "_param_metadata")
        param._param_metadata = self
        self._lr_mult: float | None = None
        self.param_type = param_type

    def features_param(
        self,
        feature_index: int,
        fptype: FeatureParamType | None = None,
        resampled=True,
        reset_optim_on_resample=True,
    ) -> Self:
        assert self.as_features is None
        self.as_features = FeaturesView(
            self.param,
            feature_index=feature_index,
            type=fptype,
            resampled=resampled,
            reset_optim_on_resample=reset_optim_on_resample,
        )
        return self

    def lr_mult(self, lr_mult: float | None = None) -> "float | None | Self":
        if lr_mult is None:
            return self._lr_mult
        self._lr_mult = lr_mult
        return self

    def has_param_group_values(self) -> bool:
        return self._lr_mult is not None

    def param_group_values(self, optim_kwargs: dict[str, Any]) -> dict[str, float]:
        # optim_kwargs is heterogeneous (e.g. lr: float, betas: tuple,
        # warmup_steps: int); only "lr" is read, and the result is lr-only.
        assert self.has_param_group_values()
        return {"lr": optim_kwargs["lr"] * self._lr_mult}


@runtime_checkable
class MetaDataParam(Protocol):
    _param_metadata: ParamMetadata
