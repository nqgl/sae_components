from typing import Any
import torch.nn as nn
from sae_components.core.cache import Cache
from sae_components.core.cache_layer import Module


class Collection(Module):
    def __init__(self, *modules, names=None, **named_modules):
        assert (len(modules) > 0) ^ (
            len(named_modules) > 0
        ), "Either unnamed or named modules should be provided, but not both"
        super().__init__()

        if len(modules) > 0:
            d = {str(i): module for i, module in enumerate(modules)}
        else:
            d = named_modules
        self._collection = nn.ModuleDict(d)


def proc_appropriately(module, x, cache, **kwargs):
    if isinstance(module, Module):
        return module(x, cache=cache, **kwargs)
    else:
        return module(x)


class Parallel(Collection):
    def __init__(self, *modules, names=None, **named_modules):
        super().__init__(*modules, names=names, **named_modules)
        self._reduction = lambda *x: x
        # ah maybe I should make this take these as list or as dict explicitly not as args and kwargs

    def forward(self, x, *, cache: Cache, **kwargs):
        l = [
            proc_appropriately(module=module, x=x, cache=cache, **kwargs)
            for i, module in enumerate(self._collection.values())
        ]
        return self._reduction(*l)

    def set_reduction(self, f):
        self._reduction = f


class Reduce: ...
