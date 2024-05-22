import torch.nn as nn
import torch
from sae_components.core.cache import Cache
from sae_components.core.module import Module


class ReuseCache(Cache):
    forward_reuse_dict: dict = ...
    sparsity_penalty: float = ...


class ReuseForward(Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, *, cache: ReuseCache, **kwargs):
        key = (id(self.module), x)
        if not cache._ancestor.has.forward_reuse_dict:
            cache._ancestor.forward_reuse_dict = {}
        elif key in cache._ancestor.forward_reuse_dict:
            print("reuse cache hit")
            return cache._ancestor.forward_reuse_dict[key]
        output = self.module(x, cache=cache, **kwargs)
        if cache._ancestor.has.forward_reuse_dict:
            cache._ancestor.forward_reuse_dict[key] = output
        else:
            print(
                "Warning: cache does not have forward_reuse_dict, so no reuse is possible. This may lead to cache overwrite errors."
            )
        return output
