import types
from collections import defaultdict
from collections.abc import Callable
from typing import (
    Literal,
    Protocol,
    TypeVar,
    overload,
)

_fields_dict: dict[type, dict[type["typeacc_method"], list[str]]] = defaultdict(dict)
# (cls -> (field_categ_name -> field_name/names))
_missing_name: set["typeacc_method"] = set()


def _getfields(cls: type, FIELD_NAME: type["typeacc_method"]) -> list[str]:
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


def getfields(cls: type, FIELD_NAME: type["typeacc_method"]) -> list[str]:
    try:
        return _getfields(cls, FIELD_NAME)
    except AttributeError:
        return []


def setfield(cls: type, FIELD_NAME: type["typeacc_method"], value: list[str]):
    assert isinstance(cls, type)
    if cls not in _fields_dict:
        _fields_dict[cls] = {}
    cls_d = _fields_dict[cls]
    cls_d[FIELD_NAME] = value


def hasfield(cls: type, FIELD_NAME: type["typeacc_method"]):
    if cls not in _fields_dict:
        return False
    cls_d = _fields_dict[cls]
    return FIELD_NAME in cls_d


class NonSingular(Protocol):
    COLLECTED_FIELD_SINGULAR: Literal[False] = False


class Singular(Protocol):
    COLLECTED_FIELD_SINGULAR: Literal[True] = True


from functools import wraps


class typeacc_method[T, **P]:
    # COLLECTED_FIELD_NAME = ...
    COLLECTED_FIELD_SINGULAR = False

    def __init__(self, func: Callable[P, T]) -> None:
        _missing_name.add(self)
        self.func = func
        self._name = None
        # update_wrapper(self, func)

    def __get__(self, instance, owner):
        return types.MethodType(self, instance)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(*args, **kwargs)

    def __set_name__(self, owner: type, name: str) -> None:
        _missing_name.remove(self)
        self._name = name
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
            setfield(owner, self.__class__, [name])

    @classmethod
    def get_fields(cls, owner: type):
        if len(_missing_name) > 0:
            raise AttributeError(
                f"some properties have not been owned: {[f.func for f in _missing_name]}"
            )
        return getfields(owner, cls)

    @overload
    @classmethod
    def get_from_fields(cls: NonSingular, inst: object) -> dict[str, Callable]: ...
    @overload
    @classmethod
    def get_from_fields(cls: Singular, inst: object) -> Callable: ...

    @classmethod
    def get_from_fields(cls, inst: object) -> dict[str, Callable] | Callable:
        fields = cls.get_fields(inst.__class__)
        assert not cls.COLLECTED_FIELD_SINGULAR
        return {f: getattr(inst, f) for f in fields}


class arch_prop_singular(typeacc_method):
    COLLECTED_FIELD_SINGULAR = True

    @classmethod
    def get_from_fields(cls, inst: object) -> Callable:
        fields = cls.get_fields(inst.__class__)
        assert cls.COLLECTED_FIELD_SINGULAR
        assert len(fields) == 1
        return getattr(inst, fields[0])


class PreForwardHook(typeacc_method): ...


def pre_forward_hook(f):
    return wraps(f)(PreForwardHook(f))


class PostForwardHook(typeacc_method): ...


def post_forward_hook(f):
    return wraps(f)(PostForwardHook(f))


class PostBackwardHook(typeacc_method): ...


def post_backward_hook(f):
    return wraps(f)(PostBackwardHook(f))


class PostStepHook(typeacc_method): ...


def post_step_hook(f):
    return wraps(f)(PostStepHook(f))
