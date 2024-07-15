from typing import Union, NewType
from pydantic import GetCoreSchemaHandler, TypeAdapter
from pydantic_core import CoreSchema, core_schema
from functools import update_wrapper
from typing import Any, Type


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
    return n * period


class RunFloat(float):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(float, handler(float))

    PERIOD_FIELD_NAME = "run_length"


class ResFloat(float):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(float, handler(float))

    PERIOD_FIELD_NAME = "resample_period"


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saeco.trainer.schedule_cfg import RunSchedulingConfig


def tosteps_wrapper(cls: Type["RunSchedulingConfig"]):
    class Class2:
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
                return getattr(self.raw, name)

    update_wrapper(Class2.__init__, cls.__init__)
    mfi = {k: v for k, v in cls.model_fields.items()}
    for name, field in mfi.items():
        annotation = cls.__annotations__[name]
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

            def get_replacements(name, t):
                @property
                def replace_field(self: Class2):
                    value = getattr(self.raw, name)
                    period = getattr(self.raw, t.PERIOD_FIELD_NAME)
                    return tosteps(value, period)

                # @replace_field.setter
                # def replace_field(self: Class2, value):
                #     setattr(self.raw, name, value)
                # no setter seems safer
                return replace_field

            setattr(Class2, name, get_replacements(name, t))
    # model_dump = cls.model_dump

    # def aliasdump(self, *a, **k):
    #     print(k)

    # cls.model_dump = aliasdump
    return Class2
