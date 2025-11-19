from collections import defaultdict
from collections.abc import Callable
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    Self,
    overload,
    runtime_checkable,
)
from warnings import deprecated

if TYPE_CHECKING:
    pass

# _T = TypeVar("_T")
_fields_dict: dict[type, dict[type["arch_prop[Any]"], list[str]]] = defaultdict(
    dict
)  # (cls -> (field_categ_name -> field_name/names))
_missing_name: set["arch_prop[Any]"] = set()
if TYPE_CHECKING:
    from saeco.components.losses import Loss


def _getfields(cls: type, FIELD_NAME: type["arch_prop[Any]"]) -> list[str]:
    if not isinstance(cls, type):
        cls = cls.__class__
    cls_d = _fields_dict[cls]
    if FIELD_NAME not in cls_d:
        for c in cls.__mro__:
            if c == cls or c is object:
                continue
            try:
                return _getfields(c, FIELD_NAME)
            except AttributeError:
                pass
        raise AttributeError(FIELD_NAME)
    return cls_d[FIELD_NAME]


def getfields(cls: type, FIELD_NAME: type["arch_prop[Any]"]) -> list[str]:
    try:
        return _getfields(cls, FIELD_NAME)
    except AttributeError:
        return []


def setfield(cls: type, FIELD_NAME: type["arch_prop[Any]"], value: list[str]):
    assert isinstance(cls, type)
    cls_d = _fields_dict[cls]
    cls_d[FIELD_NAME] = value


def hasfield(cls: type, FIELD_NAME: type["arch_prop[Any]"]):
    cls_d = _fields_dict[cls]
    return FIELD_NAME in cls_d


class NonSingular(Protocol):
    COLLECTED_FIELD_SINGULAR: Literal[False] = False


class Singular(Protocol):
    COLLECTED_FIELD_SINGULAR: Literal[True] = True


@runtime_checkable
class Instantiable(Protocol):
    _instantiated: bool

    def instantiate(self, inst_cfg: dict[str, Any] | None = None): ...


@runtime_checkable
class SetupComplete(Protocol):
    _setup_complete: Literal[True] = True


class arch_prop[T](
    cached_property,
):
    COLLECTED_FIELD_SINGULAR = False

    def __init__(self, func: Callable[[Any], T]) -> None:
        super().__init__(func)
        _missing_name.add(self)

    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> T: ...

    def __get__(self, instance: object | None, owner: Any | None = None) -> T | Self:
        if instance is not None:
            assert isinstance(instance, Instantiable) and isinstance(
                instance, SetupComplete
            )
            if not instance._instantiated:
                instance.instantiate()
            assert instance._instantiated and instance._setup_complete
        return super().__get__(instance, owner)

    def __set_name__(self, owner: type, name: str) -> None:
        _missing_name.remove(self)
        if hasfield(owner, self.__class__):
            if self.COLLECTED_FIELD_SINGULAR:
                raise AttributeError(
                    f"{self.__class__}: Cannot overwrite singular field '{name}' on {owner}"
                )
            fields = getfields(owner, self.__class__)
            fields.append(name)
            if len(fields) != len(set(fields)):
                raise AttributeError(
                    f"{self.__class__}: Field names must be unique: duplicate name '{name}' on {owner}"
                )
        else:
            # if self.COLLECTED_FIELD_SINGULAR:
            #     setfield(owner, self.__class__, name)
            # else:
            setfield(owner, self.__class__, [name])

        return super().__set_name__(owner, name)

    @classmethod
    def get_fields(cls, owner: type):
        if len(_missing_name) > 0:
            raise AttributeError(
                f"some properties have not been owned: {[f.func for f in _missing_name]}"
            )
        return getfields(owner, cls)

    @overload
    @classmethod
    def get_from_fields(cls: type[NonSingular], inst: object) -> dict[str, T]: ...
    @overload
    @classmethod
    def get_from_fields(cls: type[Singular], inst: object) -> T: ...

    @classmethod
    def get_from_fields(cls, inst: object) -> dict[str, T] | T:
        fields = cls.get_fields(inst.__class__)
        assert not cls.COLLECTED_FIELD_SINGULAR
        return {f: getattr(inst, f) for f in fields}


class arch_prop_singular[T](arch_prop[T]):
    COLLECTED_FIELD_SINGULAR = True

    @classmethod
    def get_from_fields(cls, inst: object) -> T:
        fields = cls.get_fields(inst.__class__)
        assert cls.COLLECTED_FIELD_SINGULAR
        assert len(fields) == 1
        return getattr(inst, fields[0])


import torch.nn as nn

# Loss_T = TypeVar("Loss_T", bound=nn.Module)

# from .architecture import SAE
# from saeco.components.metrics.metrics import Metric

# AuxModel_T = TypeVar("AuxModel_T", bound=nn.Module)


class loss_prop[Loss_T: nn.Module](arch_prop[Loss_T]):
    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> Loss_T: ...

    def __get__(
        self, instance: object | None, owner: Any | None = None
    ) -> Loss_T | Self:
        return super().__get__(instance, owner)


class metric_prop[Metric_T: nn.Module](arch_prop[Metric_T]):
    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> Metric_T: ...

    def __get__(
        self, instance: object | None, owner: Any | None = None
    ) -> Metric_T | Self:
        return super().__get__(instance, owner)


class _model_prop_base[T](arch_prop[T]):
    """
    Base class for model_prop and aux_model_prop.
    Adds on top of arch_prop: methods for attaching losses and metrics to a model.
    - these methods create a new loss_prop or metric_prop where
        the called function/constructor will be called with the
        model as an argument
    """

    @deprecated("define with loss_prop instead for now")
    def add_loss(self, loss: "Loss") -> None:
        raise NotImplementedError

    # # tries to infer whether this is a method (therefore needing self as the first arg)
    # # or a Loss constructor
    # from saeco.components.losses import Loss

    # assert isinstance(loss, Loss)

    # def get_loss_object(inst):
    #     return loss(self.__get__(inst))

    # return loss_prop(get_loss_object)

    # def add_metric(self, metric):
    #     def _metric(inst):
    #         return metric(self.__get__(inst))

    #     return metric_prop(_metric)


class model_prop[SAE_T: nn.Module](arch_prop_singular[SAE_T], _model_prop_base[SAE_T]):
    COLLECTED_FIELD_SINGULAR = True

    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> SAE_T: ...

    def __get__(
        self, instance: object | None, owner: Any | None = None
    ) -> SAE_T | Self:
        return super().__get__(instance, owner)

    loss = loss_prop


class aux_model_prop[AuxModel_T: nn.Module](_model_prop_base[AuxModel_T]):
    COLLECTED_FIELD_SINGULAR = False

    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(
        self, instance: object, owner: type[Any] | None = None
    ) -> AuxModel_T: ...

    def __get__(
        self, instance: object | None, owner: Any | None = None
    ) -> AuxModel_T | Self:
        return super().__get__(instance, owner)
