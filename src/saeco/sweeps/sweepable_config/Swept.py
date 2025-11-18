from pydantic import BaseModel


from typing import Any, Dict, Generic, Literal, Set, TypeVar

import pydantic._internal._model_construction as mc

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


class Swept[T](BaseModel, metaclass=SweptCheckerMeta):
    values: list[T]

    def __init__(self, *values: T, **kwargs):
        """
        Enables the Swept(a,b,c) syntax for initializing Swept fields.
        """
        if values:
            if "values" in kwargs:
                raise Exception("two sources of values in Swept initialization!")
            kwargs["values"] = values
        return super().__init__(**kwargs)

    @property
    def generic_type(self) -> type[T] | None:
        args = self.__pydantic_generic_metadata__["args"]
        if len(args) == 0:
            return None
        if len(args) > 1:
            raise Exception("Swept can only have one generic type")
        assert len(args) == 1
        return args[0]

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
