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
        if name == "_cached_calls_target":
            return super().__getattribute__(name)
        attr = getattr(self._cached_calls_target, name)
        assert callable(attr), f"Attribute {name} is not callable"

        @functools.wraps(attr)
        def wrapper(*args, **kwargs):
            stringified_call = f"{name}({args}, {kwargs})"
            if stringified_call in self._cached_calls_target.artifacts:
                return self._cached_calls_target.artifacts[stringified_call].tensor
            result = attr(*args, **kwargs)
            if isinstance(result, FilteredTensor):
                raise NotImplementedError("FilteredTensor caching not yet supported")
            assert isinstance(result, Tensor)

            self._cached_calls_target.artifacts[stringified_call] = result
            return result

        return wrapper
