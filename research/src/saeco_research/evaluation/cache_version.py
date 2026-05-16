from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def cache_version(version: Any):
    """
    Decorator attaching a cache-buster version tag to a function.

    It does not change function behavior; it only adds `._version`.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        setattr(wrapper, "_version", version)
        return wrapper

    return decorator
