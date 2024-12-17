from collections.abc import Callable
from functools import cached_property
from collections import defaultdict
from typing import Any, TYPE_CHECKING
import types

if TYPE_CHECKING:
    from .architecture import Architecture

_fields_dict = defaultdict(dict)  # (cls -> (field_categ_name -> field_name/names))
_missing_name = set()


def getfields(cls: type, FIELD_NAME):
    if not isinstance(cls, type):
        cls = cls.__class__
    cls_d = _fields_dict[cls]
    if FIELD_NAME not in cls_d:
        for c in cls.__mro__:
            if c == cls or c is object:
                continue
            try:
                return getfields(c, FIELD_NAME)
            except AttributeError:
                pass
        raise AttributeError(FIELD_NAME)
    return cls_d[FIELD_NAME]


def setfield(cls: type, FIELD_NAME, value):
    assert isinstance(cls, type)
    cls_d = _fields_dict[cls]
    cls_d[FIELD_NAME] = value


def hasfield(cls: type, FIELD_NAME):
    cls_d = _fields_dict[cls]
    return FIELD_NAME in cls_d


class arch_prop(cached_property):
    COLLECTED_FIELD_NAME = ...
    COLLECTED_FIELD_SINGULAR = False

    def __init__(self, func: Callable[[Any], Any]) -> None:
        super().__init__(func)
        _missing_name.add(self)

    def __get__(self, instance: "Architecture", owner=None):
        return super().__get__(instance, owner)

    def __set_name__(self, owner, name):
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
            if self.COLLECTED_FIELD_SINGULAR:
                setfield(owner, self.__class__, name)
            else:
                setfield(owner, self.__class__, [name])

        return super().__set_name__(owner, name)

    @classmethod
    def get_fields(cls, owner):
        if len(_missing_name) > 0:
            raise AttributeError(
                f"some properties have not been owned: {[f.func for f in _missing_name]}"
            )
        return getfields(owner, cls)

    @classmethod
    def get_from_fields(cls, inst):
        fields = cls.get_fields(inst)
        if cls.COLLECTED_FIELD_SINGULAR:
            return getattr(inst, fields)
        return {f: getattr(inst, f) for f in fields}


class _base_model_prop(arch_prop):

    def add_loss(self, fn):
        if isinstance(fn, types.FunctionType) or isinstance(fn, types.LambdaType):

            def _loss(inst):
                return fn(inst, self.__get__(inst))

        else:

            def _loss(inst):
                return fn(self.__get__(inst))

        return loss_prop(_loss)

    def add_metric(self, metric):
        def _metric(inst):
            return metric(self.__get__(inst))

        return metric_prop(_metric)


class loss_prop(arch_prop):
    def __get__(self, instance, owner=None):
        return super().__get__(instance, owner)


class metric_prop(arch_prop):
    def __get__(self, instance, owner=None):
        return super().__get__(instance, owner)


class model_prop(_base_model_prop):
    COLLECTED_FIELD_SINGULAR = True

    def __get__(self, instance, owner=None):
        return super().__get__(instance, owner)


class aux_model_prop(_base_model_prop):
    COLLECTED_FIELD_SINGULAR = False

    def __get__(self, instance, owner=None):
        return super().__get__(instance, owner)
