import pydantic._internal._model_construction as mc
from typing import (
    Literal,
    TypeVar,
    Generic,
    Type,
    List,
    Annotated,
    Union,
    Any,
    TYPE_CHECKING,
)
from pydantic import BaseModel, create_model, dataclasses


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


from typing import Set, Dict


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
    assert not isinstance(
        type, Swept
    ), "Swept type should not be wrapped in Sweepable or passed to SweepableConfig"
    return Union[Swept[type], type]


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


class ParametersForWandb(BaseModel):
    parameters: dict[str, Any]


def _to_swept_fields(target: BaseModel):
    for name, field in target.model_fields.items():
        attr = getattr(target, name)
        if isinstance(attr, Swept):
            continue
        if isinstance(attr, BaseModel):
            _to_swept_fields(attr)
            new_swept = ParametersForWandb(parameters=attr)
        else:
            new_swept = Swept[type(attr)](attr)
        setattr(target, name, new_swept)
    return target


def _to_swept_dict(target: BaseModel):
    d = {}
    for name, field in target.model_fields.items():
        attr = getattr(target, name)
        if isinstance(attr, Swept):
            subdict = attr.model_dump()
        elif isinstance(attr, BaseModel):
            subdict = dict(parameters=_to_swept_dict(attr))
        else:
            subdict = Swept[type(attr)](attr).model_dump()
        d[name] = subdict
    return d


def has_sweep(target: BaseModel):
    for name, field in target.model_fields.items():
        attr = getattr(target, name)
        if isinstance(attr, Swept):
            return True
        elif isinstance(attr, BaseModel):
            if has_sweep(attr):
                return True
    return False


def _to_swept_selective_dict(target: BaseModel):
    d = {}
    target.model_copy
    for name, field in target.model_fields.items():
        attr = getattr(target, name)
        if isinstance(attr, Swept):
            subdict = attr.model_dump()
        elif isinstance(attr, BaseModel) and has_sweep(attr):
            subdict = dict(parameters=_to_swept_selective_dict(attr))
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


if TYPE_CHECKING:
    SweepableConfig = BaseModel
else:
    ...

    class SweepableConfig(BaseModel, metaclass=SweptMeta):
        __ignore_this: int = 0

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
            return _to_swept_selective_dict(copy)
            return copy

        def from_selective_sweep(self, sweep: dict):
            mydict = self.model_dump()
            _merge_dicts_left(mydict, sweep)
            return self.model_validate(mydict)
