import pydantic._internal._model_construction as mc
from typing import TypeVar, Generic, Type, List, Annotated, Union, Any, TYPE_CHECKING
from pydantic import BaseModel, create_model, dataclasses


T = TypeVar("T")


class Swept(BaseModel, Generic[T]):
    values: list[T]

    def __init__(self, *values, **kwargs):
        if values:
            if "values" in kwargs:
                raise Exception("two sources of values in Swept initialization!")
            kwargs["values"] = values
        return super().__init__(**kwargs)


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


SweepableConfig = BaseModel
if TYPE_CHECKING:

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
