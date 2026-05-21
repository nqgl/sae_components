# %%
# %%
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING

from saeco.sweeps import SweepableConfig

if TYPE_CHECKING:
    from saeco.components.features.features_param import FeaturesParam


class OptimFieldResetValue(ABC):
    def __init__(self, num=1):
        self.num = num

    @abstractmethod
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,  # maybe add field name?
    ): ...


class ResetToConst(OptimFieldResetValue):
    def __init__(self, num=0):
        super().__init__(num)

    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        return self.num


class ResetToDirections(OptimFieldResetValue):
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        return (
            new_directions
            / new_directions.norm(dim=1, keepdim=True)
            # * mean_len
            * self.num
        )


class ResetToMean(OptimFieldResetValue):
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        if (feat_mask).all():
            return ft_optim_field_state[:].mean() * self.num
        return ft_optim_field_state[~feat_mask].mean() * self.num


class ResetToAxisMean(OptimFieldResetValue):
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        if (feat_mask).all():
            v = ft_optim_field_state[:].mean(0) * self.num
        v = ft_optim_field_state[~feat_mask].mean(0) * self.num
        assert v.ndim == 1
        assert v.shape == param.features[0].shape
        return v


class OptimFieldResetSqMeanFeatAx(OptimFieldResetValue):
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        if (feat_mask).all():
            v = ft_optim_field_state[:].pow(2).mean(0).sqrt() * self.num
        v = ft_optim_field_state[~feat_mask].pow(2).mean(0).sqrt() * self.num
        assert v.ndim == 1
        assert v.shape == param.features[0].shape
        return v


class OptimFieldResetSqMean(OptimFieldResetValue):
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        if (feat_mask).all():
            return ft_optim_field_state[:].pow(2).mean(0).sqrt() * self.num
        return ft_optim_field_state[~feat_mask].pow(2).mean(0).sqrt() * self.num


class FeatureParamType(StrEnum):
    bias = "bias"
    dec = "dec"
    enc = "enc"
    other = "other"


b2_techniques = {
    "mean": ResetToMean,
    "sq": OptimFieldResetSqMean,
    "axsq": OptimFieldResetSqMeanFeatAx,
    "axmean": ResetToAxisMean,
}


class OptimResetValuesConfig(SweepableConfig):
    optim_momentum: float = 0
    bias_momentum: float = 0
    dec_momentum: bool = False
    b2_technique: str = "sq"
    b2_scale: float = 1.0


class OptimResetValues:
    def __init__(
        self,
        cfg: OptimResetValuesConfig,
        skips=None,
        # handles: dict[str, OptimFieldResetValue] = None,
        # type_overrides: dict[
        #     FeatureParamType,
        #     dict[str, OptimFieldResetValue],
        # ] = None,
    ):
        self.skips = skips if skips is not None else {"step", "z"}

        self.handlers = {
            "momentum_buffer": ResetToConst(cfg.bias_momentum),
            "exp_avg": ResetToDirections(cfg.optim_momentum),
            "exp_avg_sq": b2_techniques[cfg.b2_technique](cfg.b2_scale),
            "mu_product": ResetToConst(0),
        }

        self.type_overrides = {
            "bias": {"exp_avg": ResetToConst()},
            "dec": {} if cfg.dec_momentum else {"exp_avg": ResetToConst(0)},
        }

    def get_value(self, field, param, ft_optim_field_state, feat_mask, new_directions):
        if (
            param.type in self.type_overrides
            and field in self.type_overrides[param.type]
        ):
            return self.type_overrides[param.type][field].get_value(
                param=param,
                ft_optim_field_state=ft_optim_field_state,
                feat_mask=feat_mask,
                new_directions=new_directions,
            )
        if field in self.handlers:
            return self.handlers[field].get_value(
                param=param,
                ft_optim_field_state=ft_optim_field_state,
                feat_mask=feat_mask,
                new_directions=new_directions,
            )
        raise KeyError(f"Field {field} not found in handlers")
