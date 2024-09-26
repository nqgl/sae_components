import functools
from typing import Any

from attrs import define, field
from torch import Tensor

from .filtered import FilteredTensor

from .metadata import Artifacts


class CachedCalls:
    def __init__(self, raw):
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
            stringified_call = f"{name}({args}, {kwargs})"
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
