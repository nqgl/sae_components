from typing import Any, overload


@overload
def useif(cond: bool, *args) -> list[Any]: ...
@overload
def useif(cond: bool, **kwargs) -> dict[str, Any]: ...


def useif(cond, *args, **kwargs) -> dict[str, Any] | list[Any] | tuple[Any]:
    assert args or kwargs and not (args and kwargs)
    if args:
        return args if cond else []
    return kwargs if cond else {}
