# %%

# %%
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from enum import Enum

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
        if (feat_mask).all():
            other = ft_optim_field_state[:]
        else:
            other = ft_optim_field_state[~feat_mask]
        # mean_len = other.norm(dim=-1).mean()
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
        assert v.ndim == 1 and v.shape == param.features[0].shape
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
        assert v.ndim == 1 and v.shape == param.features[0].shape
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


class FeatureParamType(str, Enum):
    bias = "bias"
    dec = "dec"
    enc = "enc"


from saeco.sweeps import SweepableConfig


b2_techniques = dict(
    mean=ResetToMean,
    sq=OptimFieldResetSqMean,
    axsq=OptimFieldResetSqMeanFeatAx,
    axmean=ResetToAxisMean,
)


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
        skips=set(["step"]),
        # handles: dict[str, OptimFieldResetValue] = None,
        # type_overrides: dict[
        #     FeatureParamType,
        #     dict[str, OptimFieldResetValue],
        # ] = None,
    ):
        self.skips = skips

        self.handlers = dict(
            momentum_buffer=ResetToConst(cfg.bias_momentum),
            exp_avg=ResetToDirections(cfg.optim_momentum),
            exp_avg_sq=b2_techniques[cfg.b2_technique](cfg.b2_scale),
        )

        self.type_overrides = dict(
            bias=dict(exp_avg=ResetToConst()),
            dec=dict() if cfg.dec_momentum else dict(exp_avg=ResetToConst(0)),
        )

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


# print("fields", fields)
# print(fields - self.field_handlers.skips)
# if fields & set(self.field_handlers.skips):
#     print("skipping", fields & set(self.field_handlers.skips))
#     fields = fields

# def __setitem__(self, i, v): ...

# def __getitem__(self, i): ...


# # class FPCollection


# from torchlars import LARS

# para = torch.nn.Parameter(torch.randn(20, 10))
# other_para = torch.nn.Parameter(torch.randn(20, 10))
# optim = torch.optim.RAdam([para, other_para], lr=0.01)
# # optim = LARS(optim)

# reset_features = torch.zeros(20, dtype=torch.bool)
# reset_features[2] = True
# reset_features[3] = True


# def l():
#     return (para * other_para).sum()


# # %%

# fp = FeaturesParam(para, 0)
# l().backward()
# optim.step()
# optim.zero_grad()


# def lz():
#     return (para * other_para * 0).sum()


# def test():
#     pre_p = para.clone()
#     pre_o = other_para.clone()
#     lz().backward()
#     optim.step()
#     print("para", pre_p == para)
#     print("other_para", pre_o == other_para)
#     print(
#         (
#             fp.features[reset_features]
#             == FeaturesParam(pre_p, feature_index=fp.feature_index).features[
#                 reset_features
#             ]
#         )
#         .all()
#         .item()
#     )


# m = nn.Sequential(nn.Linear(10, 10), nn.ReLU())


# test()
# # %%
# test()
# fp.reset_optim_features(optim, reset_features)
# # %%
# test()
# # optim.states
# optim.state[para].keys()
# # %%
# from saeco.core import Seq
# import torch.nn as nn


# # class FPAtest(nn.Parameter):
# #     def __init__(self, data, requires_grad, extra):
# #         super().__init__(data)
# #         print(extra)
# #         self.extra = extra

# #     @classmethod
# #     def __torch_function__(cls, func, types, args=(), kwargs=None):
# #         print(func)
# #         print(types)
# #         print(args)
# #         print([isinstance(x, cls) for x in args])

# #         if kwargs is None:
# #             kwargs = {}
# #         if func not in HANDLED_FUNCTIONS or not all(
# #             issubclass(t, (torch.Tensor, ScalarTensor)) for t in types
# #         ):
# #             return NotImplemented
# #         return HANDLED_FUNCTIONS[func](*args, **kwargs)


# # t = FPAtest(torch.ones(2), True, 2)

# # t + torch.ones(2)


# # %%


# class T:
#     def __getitem__(self, i):
#         print(i)
#         print(type(i))


# t = T()

# t[12:2]
# # %%


# class FakeParam(Tensor):
#     def __init__(
#         self,
#         param: nn.Parameter,
#         feature_index,
#         fptype: Optional[FeatureParamType] = None,
#     ):
#         super().__init__()
#         print("init")
#         self.data = param
#         self.feature_index = feature_index
#         self.field_handlers = OptimResetConfig()
#         self.resampled = False
#         self.type: Optional[FeatureParamType] = fptype and FeatureParamType(fptype)
#         print("Data", self.data)

#     def __new__(
#         cls,
#         param: nn.Parameter,
#         feature_index,
#         fptype: Optional[FeatureParamType] = None,
#     ):
#         inst = super().__new__(cls)
#         print(inst)
#         # cls.__init__(inst, param, feature_index, fptype)
#         return inst

#     def features_transform(self, tensor: Tensor) -> Tensor:
#         return tensor.transpose(0, self.feature_index)

#     def reverse_transform(self, tensor: Tensor) -> Tensor:
#         return self.features_transform(tensor)

#     def reset_optim_features(self, optim, feat_mask, new_directions=None):
#         try:
#             from torchlars import LARS

#             if isinstance(optim, LARS):
#                 optim = optim.optim
#         except ImportError:
#             pass
#         param_state = optim.state[self.param]
#         fields = set(param_state.keys())
#         print("fields", fields)
#         print(fields - self.field_handlers.skips)
#         if fields & set(self.field_handlers.skips):
#             print("skipping", fields & set(self.field_handlers.skips))
#             fields = fields - set(self.field_handlers.skips)
#         for field in fields:
#             field_state = param_state[field]
#             assert (
#                 field_state.shape == self.param.shape
#             ), f"{field}: {field_state.shape} != {self.param.shape}"
#             feat_field_state = self.features_transform(field_state)
#             feat_field_state[feat_mask] = self.field_handlers.handlers[field].get_value(
#                 param_state=param_state,
#                 param=self,
#                 ft_optim_field_state=feat_field_state,
#                 feat_mask=feat_mask,
#                 new_directions=new_directions,
#             )

#     @property
#     def features(self) -> Tensor:  # TODO is there a name with more clarity?
#         # is "data" right? thinking that might be kinda wrong
#         return self.features_transform(self.param)

#     @property
#     def grad(self) -> Tensor:
#         return self.features_transform(self.param.grad)

#     @property
#     def optimstate(self) -> OptimFieldFeatures: ...

#     def __setitem__(self, i, v): ...

#     def __getitem__(self, i): ...


# class TestPar(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fp = FakeParam(para, 0)


# tp = TestPar()
# # %%
# list(tp.parameters())
# # %%

# %%
