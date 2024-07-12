# %%

# %%
import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum


class OptimFieldResetValue(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,  # maybe add field name?
    ): ...


class OptimFieldResetToZero(OptimFieldResetValue):
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        return 0


class OptimFieldResetToNewValue(OptimFieldResetValue):
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
        mean_len = other.norm(dim=1).mean()
        return new_directions / new_directions.norm(dim=1, keepdim=True) * mean_len * 5


class OptimFieldResetToOtherMean(OptimFieldResetValue):
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        if (feat_mask).all():
            return ft_optim_field_state[:].mean()
        return ft_optim_field_state[~feat_mask].mean()


class OptimFieldResetMeanFeatAx(OptimFieldResetValue):
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        if (feat_mask).all():
            return ft_optim_field_state[:].mean(0)
        return ft_optim_field_state[~feat_mask].mean(0)


class OptimFieldResetSqMeanFeatAx(OptimFieldResetValue):
    def get_value(
        self,
        param: "FeaturesParam",
        ft_optim_field_state,
        feat_mask,
        new_directions,
    ):
        if (feat_mask).all():
            return (ft_optim_field_state[:].pow(2).mean(0)).sqrt()
        return (ft_optim_field_state[~feat_mask].pow(2).mean(0)).sqrt()


class FeatureParamType(str, Enum):
    bias = "bias"
    dec = "dec"
    enc = "enc"


class OptimResetConfig:
    def __init__(
        self,
        skips=set(["step"]),
        handles: dict[str, OptimFieldResetValue] = dict(
            momentum_buffer=OptimFieldResetToZero(),
            exp_avg=OptimFieldResetToZero(),
            exp_avg_sq=OptimFieldResetSqMeanFeatAx(),
        ),
        type_overrides: dict[
            FeatureParamType,
            dict[str, OptimFieldResetValue],
        ] = dict(
            bias=dict(exp_avg=OptimFieldResetToZero()),
            enc=dict(exp_avg=OptimFieldResetToNewValue()),
        ),
    ):
        self.skips = skips
        self.handlers = handles
        self.type_overrides = type_overrides

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


from typing import Mapping, Union, overload


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
        fptype: Optional[FeatureParamType] = None,
        resampled=True,
    ):
        super().__init__()
        self.param = param
        self.feature_index = feature_index
        self.field_handlers = OptimResetConfig()
        self.resampled = resampled
        self.type: Optional[FeatureParamType] = fptype and FeatureParamType(fptype)

    def features_transform(self, tensor: Tensor) -> Tensor:
        return tensor.transpose(0, self.feature_index)

    def reverse_transform(self, tensor: Tensor) -> Tensor:
        return self.features_transform(tensor)

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
        return self.features_transform(self.param.grad)

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
        if not self.type == "bias":
            new_directions = new_directions / new_directions.norm(dim=1, keepdim=True)
        if self.type == "enc":
            avg_other_norm = 1
            if not indices.all():
                avg_other_norm = self.features[~indices].norm(dim=1).mean()
            new_directions *= avg_other_norm * 0.2

        if self.type == FeatureParamType.bias:
            self.param[indices] = bias_reset_value
        else:
            self.features[indices] = new_directions

        optim_state = self.optimstate(optim)
        fields = set(optim_state.keys()) - set(self.field_handlers.skips)

        for field in fields:
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
            print()

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
