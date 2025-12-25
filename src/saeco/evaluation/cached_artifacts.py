import functools
import inspect
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from torch import Tensor

from saeco.data.dict_batch.dict_batch import DictBatch

from .filtered import FilteredTensor

if TYPE_CHECKING:
    from saeco.evaluation.evaluation import Evaluation


class CachedCalls:
    def __init__(self, raw: "Evaluation"):
        self.raw = raw
        assert not hasattr(self.raw, "raw")

    def __getattribute__(self, name: str) -> Any:
        if name in (
            "raw",
            "_wrap_call_with_caching____",
            "cache_some_other_call",
        ):
            return super().__getattribute__(name)
        attr = getattr(self.raw, name)
        assert callable(attr), f"Attribute {name} is not callable"
        return self._wrap_call_with_caching____(attr, name)

    def _wrap_call_with_caching____(self, func, name=None):
        name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stringified_call = f"{name}({args}, {kwargs})".replace(".", "-")
            return_type = inspect.signature(func).return_annotation
            if return_type is not None and issubclass(return_type, BaseModel):
                if self.raw.bmstore.has(func, args, kwargs):
                    return self.raw.bmstore.get(func, args, kwargs)
                value = func(*args, **kwargs)
                self.raw.bmstore.set(func, args, kwargs, value)
                return value
            if return_type is not None and issubclass(return_type, DictBatch):
                assert self.raw.artifacts.path is not None
                path = self.raw.artifacts.path / "dict_batches" / stringified_call
                if path.exists():
                    return return_type.load_from_safetensors(path)
                value = func(*args, **kwargs)
                assert isinstance(value, return_type)
                value.save_as_safetensors(path)
                return value

            if hasattr(func, "_version"):
                stringified_call += f"__v{func._version}"

            if stringified_call in self.raw.artifacts:
                return self.raw.artifacts[stringified_call]
            result = func(*args, **kwargs)
            if isinstance(result, FilteredTensor):
                raise NotImplementedError("FilteredTensor caching not yet supported")
            assert isinstance(result, Tensor)

            self.raw.artifacts[stringified_call] = result
            return result

        return wrapper

    def cache_some_other_call(self, func): ...
