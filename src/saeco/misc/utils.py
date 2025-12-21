from types import GenericAlias
from typing import Any, overload

import tqdm
from paramsight.type_utils import get_args_robust, get_origin_robust


@overload
def useif(cond: bool, *args) -> list[Any]: ...
@overload
def useif(cond: bool, **kwargs) -> dict[str, Any]: ...


def useif(cond, *args, **kwargs) -> dict[str, Any] | list[Any] | tuple[Any]:
    assert args or kwargs and not (args and kwargs)
    if args:
        return args if cond else []
    return kwargs if cond else {}


def iter_chunk_ranges(start, stop, chunk_size):  # TODO double check me
    for i in range(start, stop, chunk_size):
        yield i, min(i + chunk_size, stop)


def iter_chunk_ranges_tqdm(  # TODO double check me
    start, stop, chunk_size, measure_chunks=True, tqdm_kwargs={}
):
    if measure_chunks:
        for i in tqdm.trange(start, stop, chunk_size, **tqdm_kwargs):
            yield i, min(i + chunk_size, stop)

    else:
        for i in tqdm.tqdm(
            range(start, stop, chunk_size),
            unit_scale=chunk_size,
            total=stop - start,
            **tqdm_kwargs,
        ):
            yield i, min(i + chunk_size, stop)


def iter_chunk_qty_tqdm(
    start, stop, chunk_size, measure_chunks=True, tqdm_kwargs={}
):  # TODO double check me
    if measure_chunks:
        for i in tqdm.trange(start, stop, chunk_size, **tqdm_kwargs):
            yield min(i + chunk_size, stop) - i

    else:
        for i in tqdm.tqdm(
            range(start, stop, chunk_size),
            unit_scale=chunk_size,
            total=stop - start,
            **tqdm_kwargs,
        ):
            yield min(i + chunk_size, stop) - i


def cdiv(a, b):
    """ceiling division"""
    return (a + b - 1) // b


def assert_cast[T](tp: type[T], value: Any) -> T:
    if not isinstance(value, tp):
        raise TypeError(f"Expected {tp.__name__}, got {type(value).__name__}")
    return value


def chill_issubclass(
    cls: type | GenericAlias, target_type: type | GenericAlias
) -> bool:
    cls_t = cls if isinstance(cls, type) else get_origin_robust(cls)
    target_t = (
        target_type if isinstance(target_type, type) else get_origin_robust(target_type)
    )
    assert isinstance(cls_t, type)
    assert isinstance(target_t, type)
    if cls_t is target_t:
        if isinstance(target_type, type):
            return True
        if isinstance(cls, type):
            return False
        cls_t_params = get_args_robust(cls)
        target_t_params = get_args_robust(target_type)
        assert len(cls_t_params) == len(target_t_params)
        results = []
        for cls_t_param, target_t_param in zip(
            cls_t_params, target_t_params, strict=True
        ):
            results.append(chill_issubclass(cls_t_param, target_t_param))
        if any(results):
            assert all(results)
            return True
        return False
    return issubclass(cls_t, target_t)


if __name__ == "__main__":

    class C: ...

    class D(C): ...

    print(chill_issubclass(C | None, C))
    print(chill_issubclass(C, C))
    print(chill_issubclass(D | None, C))
    print(chill_issubclass(D | C, C))
    print(chill_issubclass(D, C))
