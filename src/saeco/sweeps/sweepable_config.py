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
from pydantic import BaseModel, create_model, dataclasses
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
            sweep_params |= attr.get_params()
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
            sweepvars |= attr.get_params()
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
    for key, value in items:
        obj_attr = getattr(obj, key)
        if isinstance(obj_attr, SweepExpression):
            orig[key] = obj_attr.instantiate()
    return orig


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

    def is_concrete(self, search_target: BaseModel = None):
        search_target = search_target or self
        for name, field in search_target.model_fields.items():
            attr = getattr(search_target, name)
            if isinstance(attr, Swept):
                return False
            elif isinstance(attr, BaseModel):
                if not self.is_concrete(attr):
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
            assert "sweep_vars" not in sweep
        mydict2 = copy.model_dump()
        _merge_dicts_left2(mydict2, sweep, copy)
        # mydict2["train_cfg"]["data_cfg"]
        return copy.model_validate(mydict2)

    def random_sweep_configuration(self):
        d = _to_randomly_selected_dict(self)
        copy = self.model_copy(deep=True)
        p = _get_sweepvars(copy)
        import random

        assert "sweep_vars" not in d
        if p:
            d["sweep_vars"] = {k.name: random.choice(k.values) for k in p}
        return self.from_selective_sweep(d)


class SweepVar(Swept[T]):
    values: list[T]
    name: str
    instantiated_value: T | None = None

    def __hash__(self):
        return id(self)

    def sweep_dump(self):
        as_swept = Swept[T](*self.values)
        return as_swept.model_dump()


class SweepExpression(Swept[T]):
    expr: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, *args, expr: Callable[[], T], **kwargs):
        super().__init__(values=[], expr=expr, args=args, kwargs=kwargs)

    def get_params(self):
        params = set()
        for p in self.args + tuple(self.kwargs.values()):
            if isinstance(p, SweepVar):
                params.add(p)
            elif isinstance(p, SweepExpression):
                params |= p.get_params()
        return params

    def get_args(self):
        return [
            a.instantiated_value if isinstance(a, SweepVar) else a for a in self.args
        ]

    def get_kwargs(self):
        return {
            k: v.instantiated_value if isinstance(v, SweepVar) else v
            for k, v in self.kwargs.items()
        }

    def instantiate(self):
        return self.expr(*self.get_args(), **self.get_kwargs())


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
