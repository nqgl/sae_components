from functools import cached_property
from types import GenericAlias, UnionType
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    Union,
    dataclass_transform,
    get_args,
    get_origin,
)

import pydantic._internal._model_construction as mc
from paramsight.generic_restored_basemodel.generic_basemodel import GenericBaseModel
from pydantic import (
    BaseModel,
)

from sweepable.has_sweep import (
    CouldHaveSweep,
    has_sweep,
    index_collection,
    key_in_collection,
    set_collection,
    to_items,
)
from sweepable.sweep_expression import SweepExpression
from sweepable.sweep_expressions import Op, SweepVar, Val
from sweepable.swept import Swept
from sweepable.swept_node import SweptNode


def generic_isinstance(inst, cls, name=None):
    try:
        return isinstance(inst, cls)
    except TypeError:
        pass
    if get_origin(cls) is Union or isinstance(cls, UnionType):
        return any(generic_isinstance(inst, a) for a in get_args(cls))
    if not isinstance(cls, GenericAlias):
        return False
    origin = get_origin(cls)
    args = get_args(cls)
    if not generic_isinstance(inst, origin):
        return False
    if origin is list:
        if not len(args) == 1:
            return False
        t = args[0]
        return all(generic_isinstance(i, t) for i in inst)
    if origin is dict:
        if not len(args) == 2:
            return False
        t_k, t_v = args
        return all(
            generic_isinstance(k, t_k) and generic_isinstance(v, t_v)
            for k, v in inst.items()
        )
    return False


NON_SWEPT_TYPES = (Literal, ClassVar)


def _to_sweepable(t, name=None):
    if t in NON_SWEPT_TYPES or get_origin(t) in NON_SWEPT_TYPES:
        return t
    if get_origin(t) is dict and get_args(t) and len(get_args(t)) == 2:
        key_type, value_type = get_args(t)
        s_t = t | dict[key_type, _to_sweepable(value_type, name=name)]
    else:
        s_t = t
    if t is ClassVar:
        return t
    assert not isinstance(t, Swept), (
        "Swept type should not be wrapped in Sweepable or passed to SweepableConfig"
    )

    from .sweep_expressions import Op

    return Union[Op[s_t], Swept[s_t], s_t]


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
                name: _to_sweepable(annotation, name=name)
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
    target: CouldHaveSweep, sweep_params: set["SweepVar"] = None
):
    d = {}
    for name, attr in to_items(target):
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


def _get_sweepvars(target: CouldHaveSweep, sweepvars: set["SweepVar"] | None = None):
    if sweepvars is None:
        sweepvars = set()
    for name, attr in to_items(target):
        if isinstance(attr, SweepExpression):
            sweepvars |= attr.get_sweepvars()
            continue
        elif isinstance(attr, CouldHaveSweep) and has_sweep(attr):
            _get_sweepvars(attr, sweepvars)
        assert not isinstance(attr, SweepVar)
    return sweepvars


def _to_randomly_selected_dict(target):
    d = {}
    import random
    import time

    random.seed(time.time())

    for name, attr in to_items(target):
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
        if key_in_collection(orig, key) and isinstance(value, CouldHaveSweep):
            obj_attr = index_collection(obj, key)
            if isinstance(obj_attr, Swept):
                # orig[key] = value
                set_collection(orig, key, value)
            else:
                set_collection(
                    orig,
                    key,
                    _merge_dicts_left2(index_collection(orig, key), value, obj_attr),
                )
        else:
            if not key_in_collection(orig, key):
                print(f"key {key} not in original dict, adding")
            set_collection(orig, key, value)
    return orig


def acc_path(obj: CouldHaveSweep, path: list[str]):
    if len(path) == 0:
        return obj
    acc = index_collection(obj, path[0])
    return acc_path(acc, path[1:])


def set_path(obj: CouldHaveSweep, path: list[str], value):
    target = acc_path(obj, path[:-1])
    key = path[-1]
    set_collection(target, key, value)


def _instantiate_sweepexpressions(target, obj: "SweepableConfig", swept_vars):
    paths = obj.sweep_info_tree.get_paths_to_sweep_expressions()
    for path in paths:
        t = acc_path(obj, path)
        assert isinstance(t, SweepExpression)
        set_path(target, path, t.evaluate(swept_vars))
        # acc_path(target, path[:-1])


def _get_sweep_expression_instantiations_dict(
    obj: "SweepableConfig", swept_vars
) -> dict:
    paths = obj.sweep_info_tree.get_paths_to_sweep_expressions()
    d = {}
    for path in paths:
        t = acc_path(obj, path)
        assert isinstance(t, SweepExpression)
        d["/".join(path)] = t.evaluate(swept_vars)
    return d


def fix_paramize(d):
    if not isinstance(d, dict):
        return d
    return {
        "parameters": {
            k: fix_paramize(v) if isinstance(v, dict) else v for k, v in d.items()
        }
    }


class SweepableConfig(GenericBaseModel, metaclass=SweepableMeta):
    """A pydantic model whose fields double as sweep specifications.

    Every field's declared type ``T`` is implicitly widened to
    ``T | Swept[T] | SweepExpression[T]``, so a sweep can be dropped in
    anywhere a plain value is expected. ``is_concrete()`` reports whether
    any sweep axes remain; ``sweep_info_tree`` describes the grid and the
    runner enumerates it into concrete instances for individual runs::

        class TrainCfg(SweepableConfig):
            lr: float = 1e-3
            pre_bias: bool = False

        TrainCfg().is_concrete()                       # True
        TrainCfg(lr=Swept(1e-3, 3e-4)).is_concrete()   # False
    """

    _ignore_this: int = 0  # needs field due to being a dataclass
    _legacy_hash_mappings: ClassVar[dict[str, str]] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__.pop("to_swept_nodes", None)
        super().__setattr__(name, value)

    def model_copy(self, *args, **kwargs) -> Self:
        copy = super().model_copy(*args, **kwargs)
        copy.__dict__.pop("to_swept_nodes", None)
        return copy

    def is_concrete(self):
        return self._is_concrete(self)

    @classmethod
    def _is_concrete(cls, search_target: BaseModel):
        for name, field in search_target.model_fields.items():
            attr = getattr(search_target, name)
            if isinstance(attr, Swept):
                return False
            elif isinstance(attr, BaseModel):
                if not cls._is_concrete(attr):
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

    def from_selective_sweep(self, sweep: dict[str, Any]) -> Self:
        sweep = sweep.copy()
        # print("sweep", sweep)
        self_copy = self.model_copy(deep=True)  # I think this is no longer needed
        p = _get_sweepvars(self_copy)
        var_data = sweep.pop("sweep_vars")
        if not p:
            #     # sv_dict = {var.name: var for var in p}
            #     # for k, v in var_data.items():
            #     #     sv_dict[k].instantiated_value = v
            # else:
            # print("no sweep vars?")
            # print("self paths", self.to_swept_nodes.get_paths_to_sweep_expressions())
            # print(
            #     "copy paths",
            #     self_copy.to_swept_nodes.get_paths_to_sweep_expressions(),
            # )

            assert "sweep_vars" not in sweep or sweep["sweep_vars"] == {}, (
                self.sweep_info_tree.get_paths_to_sweep_expressions(),
                self_copy.sweep_info_tree.get_paths_to_sweep_expressions(),
                sweep["sweep_vars"],
            )
        instantiate_dump = self_copy.model_dump()
        _merge_dicts_left2(instantiate_dump, sweep, self_copy)
        _instantiate_sweepexpressions(instantiate_dump, self_copy, var_data)
        return self_copy.model_validate(instantiate_dump)

    def get_sweepexpression_instantiations(self, sweep: dict):
        p = _get_sweepvars(self)
        if not p or "sweep_vars" not in sweep:
            return {}
        var_data = sweep.get("sweep_vars")
        return _get_sweep_expression_instantiations_dict(self, var_data)

    def random_sweep_inst_dict(self):
        return self.sweep_info_tree.random_selection()

    def random_sweep_configuration(self) -> Self:
        return self.from_selective_sweep(self.random_sweep_inst_dict())

    def select_instance_by_index(self, index: int):
        return self.from_selective_sweep(
            self.sweep_info_tree.select_instance_by_index(index)
        )

    @cached_property
    def sweep_info_tree(self) -> SweptNode:
        return SweptNode.from_sweepable(self)

    def get_hash(self) -> str:
        from hashlib import sha256

        hsh = sha256(
            self.model_dump_json(exclude_computed_fields=True).encode()
        ).hexdigest()
        if hsh in self._legacy_hash_mappings:
            return self._legacy_hash_mappings[hsh]
        return hsh
