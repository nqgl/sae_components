# %%
import torch
from torch import Tensor
from abc import ABC, abstractmethod


class OptimFieldResetValue(ABC):
    def __init__(self, field: str):
        self.field = field

    @abstractmethod
    def get_value(
        self,
        param_state,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ): ...


class OptimFieldResetToZero(OptimFieldResetValue):
    def get_value(
        self,
        param_state,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        return 0


class OptimFieldResetToOtherMean(OptimFieldResetValue):
    def get_value(
        self,
        param_state,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        if (feat_mask).all():
            return ft_optim_field_state.mean()
        return ft_optim_field_state[~feat_mask].mean()


class OptimFieldResetMeanFeatAx(OptimFieldResetValue):
    def get_value(
        self,
        param_state,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        if (feat_mask).all():
            return ft_optim_field_state.mean(0)
        return ft_optim_field_state[~feat_mask].mean(0)


class OptimFieldResetSqMeanFeatAx(OptimFieldResetValue):
    def get_value(
        self,
        param_state,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        if (feat_mask).all():
            return (ft_optim_field_state.pow(2).mean(0) + 1e-6).sqrt()
        return (ft_optim_field_state[~feat_mask].pow(2).mean(0) + 1e-6).sqrt()


class OptimResetConfig:
    def __init__(
        self,
        skips=set(["step"]),
        handles: dict[str, OptimFieldResetValue] = dict(
            momentum_buffer=OptimFieldResetToZero("momentum_buffer"),
            exp_avg=OptimFieldResetToZero("exp_avg"),
            exp_avg_sq=OptimFieldResetSqMeanFeatAx("exp_avg_sq"),
        ),
    ):
        self.skips = skips
        self.handlers = handles


class FeaturesParam:
    def __init__(self, param, feature_index=0):
        self.param = param
        self.feature_index = feature_index
        self.field_handlers = OptimResetConfig()

    def features_transform(self, tensor: Tensor) -> Tensor:
        return tensor.transpose(0, self.feature_index)

    def reverse_transform(self, tensor: Tensor) -> Tensor:
        return self.features_transform(tensor)

    def reset_optim_features(self, optim, feat_mask, new_directions=None):
        try:
            from torchlars import LARS

            if isinstance(optim, LARS):
                optim = optim.optim
        except ImportError:
            pass
        param_state = optim.state[self.param]
        fields = set(param_state.keys())
        print("fields", fields)
        print(fields - self.field_handlers.skips)
        if fields & set(self.field_handlers.skips):
            print("skipping", fields & set(self.field_handlers.skips))
            fields = fields - set(self.field_handlers.skips)
        for field in fields:
            field_state = param_state[field]
            assert (
                field_state.shape == self.param.shape
            ), f"{field}: {field_state.shape} != {self.param.shape}"
            feat_field_state = self.features_transform(field_state)
            feat_field_state[feat_mask] = self.field_handlers.handlers[field].get_value(
                param_state=param_state,
                param=self,
                ft_optim_field_state=feat_field_state,
                feat_mask=feat_mask,
                new_directions=new_directions,
            )

    @property
    def features(self) -> Tensor:
        return self.features_transform(self.param)


from torchlars import LARS

para = torch.nn.Parameter(torch.randn(20, 10))
other_para = torch.nn.Parameter(torch.randn(20, 10))
optim = torch.optim.RAdam([para, other_para], lr=0.01)
optim = LARS(optim)

reset_features = torch.zeros(20, dtype=torch.bool)
reset_features[2] = True
reset_features[3] = True


def l():
    return (para * other_para).sum()


# %%

fp = FeaturesParam(para, 0)
l().backward()
optim.step()
optim.zero_grad()


def lz():
    return (para * other_para * 0).sum()


def test():
    pre_p = para.clone()
    pre_o = other_para.clone()
    lz().backward()
    optim.step()
    print("para", pre_p == para)
    print("other_para", pre_o == other_para)
    print(
        (
            fp.features[reset_features]
            == FeaturesParam(pre_p, feature_index=fp.feature_index).features[
                reset_features
            ]
        )
        .all()
        .item()
    )


test()
# %%
test()
fp.reset_optim_features(optim, reset_features)
# %%
test()
# optim.states
optim.state[para].keys()
# %%
