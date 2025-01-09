from typing import (
    Annotated,
    Any,
    ClassVar,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
import pydantic._internal._model_construction as mc
from pydantic import BaseModel, create_model, dataclasses
from typing_extensions import dataclass_transform
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.sweeps.sweepable_config.SweptNode import SweptNode
from saeco.sweeps.sweepable_config.shared_fns import has_sweep

from saeco.sweeps.sweepable_config.sweep_expressions import SweepVar

T = TypeVar("T")


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
class SweepableMeta(mc.ModelMetaclass):
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


class SweepableConfig(BaseModel, metaclass=SweepableMeta):
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
