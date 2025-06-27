from typing import Any, overload

import tqdm


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
            **tqdm_kwargs
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
            **tqdm_kwargs
        ):
            yield min(i + chunk_size, stop) - i


def cdiv(a, b):
    """ceiling division"""
    return (a + b - 1) // b
