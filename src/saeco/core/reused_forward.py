import torch.nn as nn
import torch
from saeco.core.cache import Cache
from saeco.core.module import Module
from typing import TypeVar


class ReuseCache(Cache):
    forward_reuse_dict: dict = ...
    sparsity_penalty: float = ...


def _reuse(*, x, keyobj, callable, cache: ReuseCache, **kwargs):
    key = (id(keyobj), id(x))
    if not cache._ancestor.has.forward_reuse_dict:
        cache._ancestor.forward_reuse_dict = {}
    elif key in cache._ancestor.forward_reuse_dict:
        # print("reuse cache hit")
        return cache._ancestor.forward_reuse_dict[key]
    chk = callable
    if isinstance(callable, torch._dynamo.eval_frame.OptimizedModule):
        chk = callable._orig_mod
    if isinstance(chk, Module):
        output = callable(x, cache=cache, **kwargs)
    else:
        output = callable(x, **kwargs)
    if cache._ancestor.has.forward_reuse_dict:
        cache._ancestor.forward_reuse_dict[key] = output
    else:
        print(
            "Warning: cache does not have forward_reuse_dict, so no reuse is possible. This may lead to cache overwrite errors."
        )
    return output


def reuse_forward(*, x, module, cache: ReuseCache, **kwargs):
    return _reuse(x=x, keyobj=module, callable=module, cache=cache, **kwargs)


def reuse_method(mth):
    def wrapper(self, x: torch.Tensor, *, cache: ReuseCache, **kwargs):
        print("re_method begin")
        re = _reuse(
            x=x,
            keyobj=(self, mth),
            callable=lambda *a, **k: mth(self, *a, **k),
            cache=cache,
            **kwargs
        )
        print("re_method end")
        return re

    return wrapper


T = TypeVar("T", bound=nn.Module)


class ReuseForward(Module):
    def __init__(self, module: T):
        super().__init__()
        self.module: T = module

    def forward(self, x: torch.Tensor, *, cache: ReuseCache, **kwargs):
        return reuse_forward(x=x, module=self.module, cache=cache, **kwargs)
