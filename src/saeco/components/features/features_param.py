from saeco.components.features.optim_reset import (
    FeatureParamType,
    OptimResetValues,
)


import torch
import torch.nn as nn
from torch import Tensor
from typing import Mapping, Optional, Protocol, TypeAlias, overload, runtime_checkable

IndexType = int | list[int]


class OptimFieldFeatures:
    def __init__(
        self,
        optim: torch.optim.Optimizer,
        fp: "FeaturesParam",
        field: Optional[str] = None,
        index=None,
    ):
        self.optim = optim
        self.fp = fp
        self.index = index
        self.field = field

    @overload
    def __getitem__(self, i: IndexType) -> Mapping[str, Tensor]: ...
    @overload
    def __getitem__(self, i: str) -> Mapping[IndexType, Tensor]: ...

    def parse_index_field(self, i):
        if isinstance(i, tuple) and any([isinstance(el, str) for el in i]):
            assert len(i) == 2
            if isinstance(i[0], str):
                field, index = i
            else:
                index, field = i
        elif isinstance(i, str):
            field = i
            index = None
        else:
            index = i
            field = None
        if index is not None and self.index is not None:
            raise KeyError("Index already set")
        if field is not None and self.field is not None:
            raise KeyError("Field already set")

        index = index if index is not None else self.index
        field = field if field is not None else self.field
        return index, field

    def __getitem__(self, i: tuple) -> Tensor:
        index, field = self.parse_index_field(i)
        if None in (index, field):
            return OptimFieldFeatures(self.optim, self.fp, field, index)
        return self.fp.features_transform(self.pstate[field])[index]

    def __setitem__(self, i, v):
        index, field = self.parse_index_field(i)
        if None in (index, field):
            raise KeyError("Index or field not set")
        self.fp.features_transform(self.pstate[field])[index] = v

    @property
    def pstate(self) -> Mapping[str, Tensor]:
        return self.optim.state[self.fp.param]

    def keys(self):
        return self.pstate.keys()


class FeaturesParam:
    def __init__(
        self,
        param: nn.Parameter,
        feature_index,
        feature_parameter_type: Optional[FeatureParamType] = None,
        resampled=True,
        reset_optim_on_resample=True,
    ):
        super().__init__()
        self.param = param
        self.feature_index = feature_index
        self.field_handlers = None
        self.resampled = resampled
        self.type: Optional[FeatureParamType] = (
            feature_parameter_type and FeatureParamType(feature_parameter_type)
        )
        self.resampler_cfg = None
        self.reset_optim_on_resample = reset_optim_on_resample

    def __eq__(self, other):
        return (
            self.param is other.param
            and self.feature_index == other.feature_index
            and self.field_handlers == other.field_handlers
            and self.resampled == other.resampled
            and self.type == other.type
            and self.resampler_cfg == other.resampler_cfg
        )

    def __hash__(self):
        return hash(
            (
                self.param,
                self.feature_index,
                self.field_handlers,
                self.resampled,
                self.type,
            )
        )

    def set_cfg(self, cfg):
        if self.resampler_cfg is cfg:
            return
        self.field_handlers = OptimResetValues(cfg.optim_reset_cfg)
        self.resampler_cfg = cfg

    def features_transform(self, tensor: Tensor) -> Tensor:
        return tensor.transpose(0, self.feature_index)

    def reverse_transform(self, tensor: Tensor) -> Tensor:
        return self.features_transform(tensor)

    FPTYPES: type[FeatureParamType] = FeatureParamType
    # def reset_optim_features(self, optim, feat_mask, new_directions=None):
    #     try:
    #         from torchlars import LARS

    #         if isinstance(optim, LARS):
    #             optim = optim.optim
    #     except ImportError:
    #         pass
    #     param_state = optim.state[self.param]
    #     fields = set(param_state.keys())
    #     print("fields", fields)
    #     print(fields - self.field_handlers.skips)
    #     if fields & set(self.field_handlers.skips):
    #         print("skipping", fields & set(self.field_handlers.skips))
    #         fields = fields - set(self.field_handlers.skips)
    #     for field in fields:
    #         field_state = param_state[field]
    #         assert (
    #             field_state.shape == self.param.shape
    #         ), f"{field}: {field_state.shape} != {self.param.shape}"
    #         feat_field_state = self.features_transform(field_state)
    #         feat_field_state[feat_mask] = self.field_handlers.handlers[field].get_value(
    #             param_state=param_state,
    #             param=self,
    #             ft_optim_field_state=,
    #             feat_mask=feat_mask,
    #             new_directions=new_directions,
    #         )

    @property
    def features(self) -> Tensor:  # TODO is there a name with more clarity?
        # is "data" right? thinking that might be kinda wrong
        return self.features_transform(self.param)

    @property
    def grad(self) -> Tensor:
        if self.param.grad is None:
            return None
        return self.features_transform(self.param.grad)

    def get_optim(self, optim):
        try:
            from torchlars import LARS

            if isinstance(optim, LARS):
                optim = optim.optim
        except ImportError:
            pass
        return optim

    def optimstate(self, optim) -> OptimFieldFeatures:
        try:
            from torchlars import LARS

            if isinstance(optim, LARS):
                optim = optim.optim
        except ImportError:
            pass
        return OptimFieldFeatures(optim, self)

    @torch.no_grad()
    def resample(
        self, *, indices, new_directions, bias_reset_value, optim: torch.optim.Optimizer
    ):
        if isinstance(new_directions, tuple):
            assert len(new_directions) == 2
            if self.type == "enc":
                new_directions = new_directions[0]
            elif self.type == "dec":
                new_directions = new_directions[1]
            elif self.type == "other":
                new_directions = new_directions[0]
        if not self.type == "bias":
            new_directions = new_directions / new_directions.norm(dim=1, keepdim=True)
        if self.type == "enc":
            avg_other_norm = 1
            if not indices.all():
                avg_other_norm = self.features[~indices].norm(dim=1).mean()
            new_directions *= avg_other_norm * 0.2

        if self.type == FeatureParamType.bias:
            self.features[indices] = bias_reset_value
        else:
            self.features[indices] = new_directions
        if not self.reset_optim_on_resample:
            return
        optim_state = self.optimstate(optim)
        fields = set(optim_state.keys()) - set(self.field_handlers.skips)

        for field in fields:
            if self.get_optim(optim).state[field] == {}:
                continue
            field_state = optim_state[field]
            assert (
                field_state[:].shape == self.features.shape
            ), f"{field}: {field_state[:].shape} != {self.features.shape}"

            optim_state[field, indices] = self.field_handlers.get_value(
                field=field,
                param=self,
                ft_optim_field_state=optim_state[field],
                feat_mask=indices,
                new_directions=new_directions,
            )


def get_featuresparams(model: nn.Module) -> set[FeaturesParam]:
    l: set[FeaturesParam] = set()
    for m in model.modules():
        if isinstance(m, HasFeatures):
            l |= set(m.features.values())
    d = {}
    for fp in l:
        if fp.param in d:
            other = d[fp.param]
            assert other == fp, f"{other} != {fp}"
            raise ValueError(
                f"Duplicate feature parameter {fp}. implement __eq__ and change this check to just (intelligently) deduplicate and check for inconsistency"
            )
        d[fp.param] = fp
    return l


def get_resampled_params(model: nn.Module):
    for fp in get_featuresparams(model):
        if fp.resampled:
            yield fp


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

from typing import overload
from functools import cached_property


@runtime_checkable
class HasFeaturesAttr(Protocol):
    features: dict[str, FeaturesParam]


@runtime_checkable
class HasFeaturesCachedProperty(Protocol):
    @cached_property
    def features(self) -> dict[str, FeaturesParam]: ...


@runtime_checkable
class HasFeaturesProperty(Protocol):

    @property
    def features(self) -> dict[str, FeaturesParam]: ...


HasFeatures: TypeAlias = (
    HasFeaturesProperty | HasFeaturesCachedProperty | HasFeaturesAttr
)
