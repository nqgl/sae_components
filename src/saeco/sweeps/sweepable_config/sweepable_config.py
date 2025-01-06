import random
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Generic,
    List,
    Literal,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import pydantic._internal._model_construction as mc
from pydantic import BaseModel, create_model, dataclasses, Field
from typing_extensions import dataclass_transform

T = TypeVar("T")


class SweptCheckerMeta(mc.ModelMetaclass):
    def __instancecheck__(self, instance: Any) -> bool:
        if mc.ModelMetaclass.__instancecheck__(self, instance):
            return True
        if self is Swept:
            return False
        if not isinstance(instance, Swept):
            return False
        iT = instance.__pydantic_generic_metadata__["args"]
        sT = self.__pydantic_generic_metadata__["args"]
        if len(iT) == len(sT) == 1:
            try:
                if issubclass(iT[0], sT[0]):
                    return True
            except TypeError:
                pass
        if len(sT) == 1 and all(isinstance(v, sT) for v in instance.values):
            return True
        return False


from typing import Dict, Set


class Swept(BaseModel, Generic[T], metaclass=SweptCheckerMeta):
    values: list[T]

    def __init__(self, *values, **kwargs):
        if values:
            if "values" in kwargs:
                raise Exception("two sources of values in Swept initialization!")
            kwargs["values"] = values
        return super().__init__(**kwargs)

    def model_dump(
        self,
        *,
        mode: str = "python",
        include: Set[int] | Set[str] | Dict[int, Any] | Dict[str, Any] | None = None,
        exclude: Set[int] | Set[str] | Dict[int, Any] | Dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none"] | Literal["warn"] | Literal["error"] = True,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        dump = super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )
        # T_args = self.__pydantic_generic_metadata__["args"]
        # if T_args:
        for i, v in enumerate(dump["values"]):
            if isinstance(v, bool):
                dump["values"][i] = int(v)

        return dump


def Sweepable(type):
    if type is ClassVar:
        return type
    assert not isinstance(
        type, Swept
    ), "Swept type should not be wrapped in Sweepable or passed to SweepableConfig"
    return Union[Swept[type], type]


@dataclass_transform(
    kw_only_default=True,
)
class SweptMeta(mc.ModelMetaclass):
    def __new__(
        mcs,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        __pydantic_generic_metadata__=None,
        __pydantic_reset_parent_namespace__: bool = True,
        _create_model_module: str | None = None,
        **kwargs: Any,
    ) -> type:
        if "__annotations__" in namespace:
            namespace["__annotations__"] = {
                name: Sweepable(annotation)
                for name, annotation in namespace["__annotations__"].items()
            }
        return super().__new__(
            mcs,
            cls_name,
            bases,
            namespace,
            __pydantic_generic_metadata__,
            __pydantic_reset_parent_namespace__,
            _create_model_module,
            **kwargs,
        )


# class ParametersForWandb(BaseModel):
#     parameters: dict[str, Any]


# def _to_swept_fields(target: BaseModel):
#     for name, field in target.model_fields.items():
#         attr = getattr(target, name)
#         if isinstance(attr, Swept):
#             continue
#         if isinstance(attr, BaseModel):
#             _to_swept_fields(attr)
#             new_swept = ParametersForWandb(parameters=attr)
#         else:
#             new_swept = Swept[type(attr)](attr)
#         setattr(target, name, new_swept)
#     return target


# def _to_swept_dict(target: BaseModel):
#     d = {}
#     for name, field in target.model_fields.items():
#         attr = getattr(target, name)
#         if isinstance(attr, Swept):
#             subdict = attr.model_dump()
#         elif isinstance(attr, BaseModel):
#             subdict = dict(parameters=_to_swept_dict(attr))
#         else:
#             subdict = Swept[type(attr)](attr).model_dump()
#         d[name] = subdict
#     return d


def has_sweep(target: BaseModel | dict):
    if isinstance(target, BaseModel):
        items = [(k, getattr(target, k)) for (k, v) in target.model_fields.items()]
    else:
        assert isinstance(target, dict)
        items = target.items()
    for name, attr in items:
        if isinstance(attr, Swept):
            return True
        elif isinstance(attr, BaseModel | dict):
            if has_sweep(attr):
                return True
    return False


from functools import singledispatch


def _to_swept_selective_dict(
    target: BaseModel | dict, sweep_params: set["SweepVar"] = None
):
    d = {}
    if isinstance(target, BaseModel):
        items = [(k, getattr(target, k)) for (k, v) in target.model_fields.items()]
    else:
        assert isinstance(target, dict)
        items = target.items()
    for name, attr in items:
        if isinstance(attr, SweepExpression):
            sweep_params |= attr.get_sweepvars()
            continue
        elif isinstance(attr, SweepVar):
            sweep_params.add(attr)
            continue
        elif isinstance(attr, Swept):
            subdict = attr.model_dump()
        elif isinstance(attr, BaseModel | dict) and has_sweep(attr):
            subdict = dict(parameters=_to_swept_selective_dict(attr, sweep_params))
        elif isinstance(attr, dict):
            print("dict at", name, attr)
            continue
        else:
            continue
        d[name] = subdict
    return d


def _get_sweepvars(target: BaseModel | dict, sweepvars: set["SweepVar"] = None):
    if isinstance(target, BaseModel):
        items = [(k, getattr(target, k)) for (k, v) in target.model_fields.items()]
    else:
        assert isinstance(target, dict)
        items = target.items()
    if sweepvars is None:
        sweepvars = set()
    for name, attr in items:
        if isinstance(attr, SweepExpression):
            sweepvars |= attr.get_sweepvars()
            continue
        elif isinstance(attr, SweepVar):
            assert False
            sweepvars.add(attr)
            continue
        elif isinstance(attr, BaseModel | dict) and has_sweep(attr):
            _get_sweepvars(attr, sweepvars)
    return sweepvars


def _to_randomly_selected_dict(target):
    d = {}
    import random
    import time

    random.seed(time.time())

    if isinstance(target, BaseModel):
        items = [(k, getattr(target, k)) for (k, v) in target.model_fields.items()]
    else:
        assert isinstance(target, dict)
        items = target.items()
    for name, attr in items:
        if isinstance(attr, Swept):
            subdict = random.choice(attr.model_dump()["values"])
        elif isinstance(attr, BaseModel | dict):
            subdict = _to_randomly_selected_dict(attr)
        else:
            continue
        d[name] = subdict
    return d


def _merge_dicts_left(orig, new):
    for key, value in new.items():
        if key in orig and isinstance(value, dict):
            orig[key] = _merge_dicts_left(orig[key], value)
        else:
            if key not in orig:
                print(f"key {key} not in original dict, adding")
            orig[key] = value
    return orig


def _merge_dicts_left2(orig, new, obj):
    for key, value in new.items():
        if key in orig and isinstance(value, dict):
            obj_attr = getattr(obj, key)
            if isinstance(obj_attr, Swept):
                orig[key] = value
            else:
                orig[key] = _merge_dicts_left2(orig[key], value, obj_attr)
        else:
            if key not in orig:
                print(f"key {key} not in original dict, adding")
            orig[key] = value
    if isinstance(obj, BaseModel):
        items = [(k, getattr(obj, k)) for (k, v) in obj.model_fields.items()]
    # elif isinstance(obj, dict):
    #     items = obj.items()
    else:
        return orig
    # for key, value in items:
    #     obj_attr = getattr(obj, key)
    #     if isinstance(obj_attr, SweepExpression):
    #         orig[key] = obj_attr.instantiate()
    return orig


def acc_path(obj, path: list[str]):
    if len(path) == 0:
        return obj
    acc = obj[path[0]] if isinstance(obj, dict) else getattr(obj, path[0])
    return acc_path(acc, path[1:])


def set_path(obj, path: list[str], value):
    target = acc_path(obj, path[:-1])
    key = path[-1]
    if isinstance(target, dict):
        target[key] = value
    else:
        setattr(target, key, value)


def _instantiate_sweepexpressions(target, obj: "SweepableConfig", swept_vars):
    paths = obj.to_swept_nodes().get_paths_to_sweep_expressions()
    for path in paths:
        t: SweepExpression = acc_path(obj, path)
        set_path(target, path, t.instantiated(swept_vars))
        # acc_path(target, path[:-1])


def fix_paramize(d):
    if not isinstance(d, dict):
        return d
    return {
        "parameters": {
            k: fix_paramize(v) if isinstance(v, dict) else v for k, v in d.items()
        }
    }


class SweepableConfig(BaseModel, metaclass=SweptMeta):
    _ignore_this: int = 0  # needs field due to being a dataclass

    def is_concrete(self):
        return self._is_concrete(self)

    @classmethod
    def _is_concrete(cls, search_target: BaseModel):
        for name, field in search_target.model_fields.items():
            attr = getattr(search_target, name)
            if isinstance(attr, Swept):
                return False
            elif isinstance(attr, BaseModel):
                if not cls.is_concrete(attr):
                    return False
        return True

    def sweep(self):
        copy = self.model_copy(deep=True)
        d = _to_swept_selective_dict(copy)
        p = _get_sweepvars(copy)
        sv_dict = {k.name: k.sweep_dump() for k in p}
        assert "sweep_vars" not in d
        if sv_dict:
            d["sweep_vars"] = {"parameters": sv_dict}
        return d

    def from_selective_sweep(self, sweep: dict):
        # mydict = self.model_dump()
        # _merge_dicts_left(mydict, sweep)
        copy = self.model_copy(deep=True)
        p = _get_sweepvars(copy)
        if p:
            var_data = sweep.pop("sweep_vars")
            sv_dict = {var.name: var for var in p}
            for k, v in var_data.items():
                sv_dict[k].instantiated_value = v
        else:
            print("no sweep vars?")
            print("self paths", self.to_swept_nodes().get_paths_to_sweep_expressions())
            print("copy paths", copy.to_swept_nodes().get_paths_to_sweep_expressions())

            assert "sweep_vars" not in sweep, (
                self.to_swept_nodes().get_paths_to_sweep_expressions(),
                copy.to_swept_nodes().get_paths_to_sweep_expressions(),
            )
        mydict2 = copy.model_dump()
        _merge_dicts_left2(mydict2, sweep, copy)
        _instantiate_sweepexpressions(mydict2, copy, var_data)
        # mydict2["train_cfg"]["data_cfg"]
        return copy.model_validate(mydict2)

    def random_sweep_inst_dict(self):
        d = _to_randomly_selected_dict(self)
        copy = self.model_copy(deep=True)
        p = _get_sweepvars(copy)
        import random

        assert "sweep_vars" not in d
        if p:
            d["sweep_vars"] = {k.name: random.choice(k.values) for k in p}
        return d

    def random_sweep_configuration(self):
        return self.from_selective_sweep(self.random_sweep_inst_dict())

    def to_swept_nodes(self):
        return SweptNode.from_sweepable(self)


class SweepVar(Swept[T]):
    values: list[T]
    name: str
    instantiated_value: T | None = None

    def __hash__(self):
        return hash(self.name)

    def sweep_dump(self):
        as_swept = Swept[T](*self.values)
        return as_swept.model_dump()


class SweepExpression(Swept[T]):
    expr: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, *args, expr: Callable[[], T], **kwargs):
        super().__init__(values=[], expr=expr, args=args, kwargs=kwargs)

    def get_sweepvars(self) -> set["SweepVar"]:
        params = set()
        for p in self.args + tuple(self.kwargs.values()):
            if isinstance(p, SweepVar):
                params.add(p)
            elif isinstance(p, SweepExpression):
                params |= p.get_sweepvars()
        return params

    def evaluate(self, *args, **kwargs):
        print(f"evaluating SweepExpression {self.expr}(*{args}, **{kwargs})")
        return self.expr(*args, **kwargs)

    def instantiate(self):
        return self.evaluate(*self.get_args(), **self.get_kwargs())

    def instantiated(self, swept_vars):
        return self.evaluate(*self.get_args(swept_vars), **self.get_kwargs(swept_vars))

    def get_args(self, swept_vars=None):
        if swept_vars is None:
            return [
                a.instantiated_value if isinstance(a, SweepVar) else a
                for a in self.args
            ]
        return [
            swept_vars[arg.name] if isinstance(arg, SweepVar) else arg
            for arg in self.args
        ]

    def get_kwargs(self, swept_vars=None):
        if swept_vars is None:
            return {
                k: v.instantiated_value if isinstance(v, SweepVar) else v
                for k, v in self.kwargs.items()
            }
        return {
            k: swept_vars[kwarg.name] if isinstance(kwarg, SweepVar) else kwarg
            for k, kwarg in self.kwargs.items()
        }


Location = list[str]


class SweptNode(BaseModel):
    location: Location = Field(default_factory=list)
    children: dict[str, "SweptNode"] = Field(default_factory=dict)
    swept_fields: dict[str, Swept] = Field(default_factory=dict)
    expressions: dict[str, SweepExpression] = Field(default_factory=dict)
    # sweepvars: set[SweepVar] = Field(default_factory=set)

    @classmethod
    def from_sweepable(
        cls,
        target: BaseModel | dict,
        location: Location = [],
    ) -> "SweptNode":
        inst = cls(location=location)
        if isinstance(target, BaseModel):
            items = [(k, getattr(target, k)) for (k, v) in target.model_fields.items()]
        else:
            assert isinstance(target, dict)
            items = target.items()
        for name, attr in items:
            if isinstance(attr, SweepExpression):
                inst.expressions[name] = attr
                # inst.sweepvars |= attr.get_sweepvars()
            elif isinstance(attr, SweepVar):
                raise NotImplementedError("sweepvars should be inside an expression")
                # inst.sweepvars.add(attr)
            elif isinstance(attr, Swept):
                inst.swept_fields[name] = attr
            elif isinstance(attr, BaseModel | dict) and has_sweep(attr):
                inst.children[name] = cls.from_sweepable(attr, location + [name])
            else:
                continue

        return inst

    def get_sweepvars(self) -> set[SweepVar]:
        s = set()
        for k, v in self.expressions.items():
            s |= v.get_sweepvars()
        for k, v in self.children.items():
            s |= v.get_sweepvars()
        return s

    def swept_combination(self):
        """excluding sweepvar options"""
        n = 1
        for k, v in self.swept_fields.items():
            n *= len(v.values)
        for k, v in self.children.items():
            n *= v.swept_option_count()
        return n

    def swept_options_sum(self):
        n = 0
        for k, v in self.swept_fields.items():
            n += len(v.values)
        for k, v in self.children.items():
            n += v.swept_options_sum()
        return n

    def to_wandb(self):
        sweepvars = self.get_sweepvars()
        return {
            "parameters": {
                "sweep_vars": {
                    "parameters": {k.name: k.sweep_dump() for k in sweepvars}
                },
                **{
                    k: v._to_wandb_parameters_only()
                    for k, v in self.children.items()
                    if v.swept_options_sum() > 0
                },
                **{k: v.model_dump() for k, v in self.swept_fields.items()},
            },
            "method": "grid",
        }

    def _to_wandb_parameters_only(self):
        return {
            "parameters": {
                **{k: v._to_wandb_parameters_only() for k, v in self.children.items()},
                **{k: v.model_dump() for k, v in self.swept_fields.items()},
            }
        }

    def random_selection(self, sweep_vars=None):
        if sweep_vars is None:
            vars = self.get_sweepvars()
            var_values = {var.name: random.choice(var.values) for var in vars}
            return {
                **self.random_selection(var_values),
                "sweep_vars": var_values,
            }
        return {
            **{k: v.random_selection(sweep_vars) for k, v in self.children.items()},
            **{k: random.choice(v.values) for k, v in self.swept_fields.items()},
        }

    def get_paths_to_sweep_expressions(self) -> list[Location]:
        paths = []
        for k, v in self.children.items():
            paths.extend([[k] + p for p in v.get_paths_to_sweep_expressions()])
        paths.extend([[k] for k in self.expressions.keys()])
        return paths


def add_dictlist(d1: dict[Any, list], d2: dict[Any, list]) -> dict[Any, list]:
    out = {}
    k1 = set(d1.keys())
    k2 = set(d2.keys())
    for k in k1 & k2:
        out[k] = d1[k] + d2[k]
    for k in k1 - k2:
        out[k] = d1[k].copy()
    for k in k2 - k1:
        out[k] = d2[k].copy()
    return out


def test():
    class Test(SweepableConfig):
        x: int
        y: int

    class Nest(SweepableConfig):
        t: Test
        z: int
        e: int

    n = Nest(t=Test(x=1, y=Swept(2, 3)), z=3, e=Swept(1, 2))
    print(n.sweep())
    t = Test(x=Swept(1, 2), y=2)
    print(t.sweep())

    class SubclassOfSwept(Swept):
        def blah(self):
            pass

    t2 = Test(x=SubclassOfSwept(1, 2), y=2)
    print(t2.sweep())

    sp = SweepVar(1, 2, 3, name="var1")
    t3 = Test(
        x=SweepExpression(sp, expr=lambda x: x),
        y=SweepExpression(sp, expr=lambda x: x + 1),
    )
    print(t3.sweep())


if __name__ == "__main__":
    test()
