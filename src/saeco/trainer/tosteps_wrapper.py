from functools import update_wrapper
from typing import Any, NewType, Type, Union

from pydantic import GetCoreSchemaHandler, TypeAdapter
from pydantic_core import core_schema, CoreSchema


def tosteps(n: int | float, period: int | None = None) -> int:
    # some values will be expressed as either
    # a number of steps
    # or a fraction of some period, default run length
    # signified by type -- ints are steps, floats are proportions
    # this converts proportions to steps and leaves steps as is
    assert 0 <= n
    if isinstance(n, int):
        return n
    assert isinstance(n, float) and n <= 1 and isinstance(period, int)
    return int(n * period)


class FloatCheckMeta(type):
    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, float):
            return True
        return super().__instancecheck__(__instance)


class RunFloat(float, metaclass=FloatCheckMeta):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(float, handler(float))

    PERIOD_FIELD_NAME = "run_length"


class ResFloat(float, metaclass=FloatCheckMeta):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(float, handler(float))

    PERIOD_FIELD_NAME = "resample_period"


from typing import TYPE_CHECKING, Annotated, get_origin, get_args

if TYPE_CHECKING:
    from saeco.trainer.schedule_cfg import RunSchedulingConfig


def deannotate(annotation):
    if isinstance(annotation, Annotated):
        return get_args(annotation)[0]
    return annotation


def tosteps_wrapper(cls: Type["RunSchedulingConfig"]):
    class Class2:
        _IS_WRAPPED = True

        def __init__(
            self,
            *a,
            _raw=None,
            **k,
        ):
            self.raw = _raw or cls(*a, **k)

        def __getattribute__(self, name: str) -> Any:
            try:
                return super().__getattribute__(name)
            except AttributeError:
                if name in cls.__dict__:
                    v = cls.__dict__[name]
                    if callable(v):
                        return lambda *a, **k: v(self, *a, **k)
                    elif isinstance(v, property):
                        return v.fget(self)

                return getattr(self.raw, name)

    update_wrapper(Class2.__init__, cls.__init__)
    mfi = {k: v for k, v in cls.model_fields.items()}

    def get_replacements(name, t):
        @property
        def replace_field(self: Class2):
            value = getattr(self.raw, name)
            period = getattr(self.raw, t.PERIOD_FIELD_NAME)
            return tosteps(value, period)

        return replace_field

    for name, field in mfi.items():
        annotation = deannotate(field.annotation)
        if issubclass(int, annotation):
            if issubclass(RunFloat, annotation) and issubclass(ResFloat, annotation):
                raise Exception(
                    "Warning: both RunFloat and ResFloat, skipping. <int | float> type will not be replaced"
                )
            elif issubclass(RunFloat, annotation):
                t = RunFloat
                print("found run", name)
            elif issubclass(ResFloat, annotation):
                t = ResFloat
                print("found res", name)
            else:
                continue

                # @replace_field.setter
                # def replace_field(self: Class2, value):
                #     setattr(self.raw, name, value)
                # no setter seems safer

            setattr(Class2, name, get_replacements(name, t))
    # model_dump = cls.model_dump

    # def aliasdump(self, *a, **k):
    #     print(k)

    # cls.model_dump = aliasdump
    return Class2
