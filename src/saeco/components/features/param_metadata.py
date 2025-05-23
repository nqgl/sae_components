import torch
import torch.nn as nn
from typing import Protocol, runtime_checkable, Union
from .features_param import FeaturesParam, FeatureParamType


class ParamMetadata:
    def __init__(self, param, param_type=None):
        self.param = param
        self.as_features: FeaturesParam | None = None
        assert not hasattr(param, "_param_metadata")
        param._param_metadata = self
        self._lr_mult = None
        self.param_type = param_type

    def features_param(
        self,
        feature_index,
        fptype: FeatureParamType | None = None,
        resampled=True,
        reset_optim_on_resample=True,
    ) -> "ParamMetadata":
        assert self.as_features is None
        self.as_features = FeaturesParam(
            self.param,
            feature_index=feature_index,
            feature_parameter_type=fptype,
            resampled=resampled,
            reset_optim_on_resample=reset_optim_on_resample,
        )
        return self

    def lr_mult(self, lr_mult=None) -> Union[float, "ParamMetadata"]:
        if lr_mult is None:
            return self._lr_mult
        self._lr_mult = lr_mult
        return self

    def has_param_group_values(self) -> bool:
        return self._lr_mult is not None

    def param_group_values(self, optim_kwargs: dict) -> dict:
        assert self.has_param_group_values()
        return {"lr": optim_kwargs["lr"] * self._lr_mult}


@runtime_checkable
class MetaDataParam(Protocol):
    _param_metadata: ParamMetadata
