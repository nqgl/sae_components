import functools
from typing import Any

from attrs import define, field
from torch import Tensor

from .filtered import FilteredTensor

from .metadata import Artifacts


@define
class CachedCalls:
    _cached_calls_target = field()

    def __getattribute__(self, name: str) -> Any:
        if name in (
            "_cached_calls_target",
            "_wrap_call_with_caching____",
            "cache_some_other_call",
        ):
            return super().__getattribute__(name)
        attr = getattr(self._cached_calls_target, name)
        assert callable(attr), f"Attribute {name} is not callable"
        return self._wrap_call_with_caching____(attr, name)

    def _wrap_call_with_caching____(self, func, name=None):
        name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stringified_call = f"{name}({args}, {kwargs})"
            if stringified_call in self._cached_calls_target.artifacts:
                return self._cached_calls_target.artifacts[stringified_call]
            result = func(*args, **kwargs)
            if isinstance(result, FilteredTensor):
                raise NotImplementedError("FilteredTensor caching not yet supported")
            assert isinstance(result, Tensor)

            self._cached_calls_target.artifacts[stringified_call] = result
            return result

        return wrapper

    def cache_some_other_call(self, func): ...
