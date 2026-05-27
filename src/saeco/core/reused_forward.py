import torch
import torch.nn as nn

from saeco.components.wrap import WrapsModule
from saeco.core.cache import Cache
from saeco.core.module import Module


class ReuseCache(Cache):
    forward_reuse_dict: dict = ...
    sparsity_penalty: float = ...


def _reuse(*, x, keyobj, func, cache: ReuseCache, **kwargs):
    key = (id(keyobj), id(x))
    if not cache._ancestor.has.forward_reuse_dict:
        cache._ancestor.forward_reuse_dict = {}
    elif key in cache._ancestor.forward_reuse_dict:
        return cache._ancestor.forward_reuse_dict[key]
    chk = func
    if isinstance(func, torch._dynamo.eval_frame.OptimizedModule):
        chk = func._orig_mod
    if isinstance(chk, Module):
        output = func(x, cache=cache, **kwargs)
    else:
        output = func(x, **kwargs)
    if cache._ancestor.has.forward_reuse_dict:
        cache._ancestor.forward_reuse_dict[key] = output
    else:
        print(
            "Warning: cache does not have forward_reuse_dict, so no reuse is possible. "
            "This may lead to cache overwrite errors."
        )
    return output


def reuse_forward(*, x, module, cache: ReuseCache, **kwargs):
    return _reuse(x=x, keyobj=module, func=module, cache=cache, **kwargs)


def reuse_method(mth):
    def wrapper(self, x: torch.Tensor, *, cache: ReuseCache, **kwargs):
        re = _reuse(
            x=x,
            keyobj=(self, mth),
            func=lambda *a, **k: mth(self, *a, **k),
            cache=cache,
            **kwargs,
        )
        return re

    return wrapper


class ReuseForward[T: Module](WrapsModule[T]):
    def __init__(self, module: T):
        super().__init__(module)

    def __call__(self, x: torch.Tensor, *, cache: ReuseCache, **kwargs):
        return reuse_forward(x=x, module=self.__wrapped__, cache=cache, **kwargs)


Module.register(ReuseForward)
