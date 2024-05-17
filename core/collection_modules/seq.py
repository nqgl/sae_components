from typing import Any
import torch.nn as nn
from sae_components.core import CacheModule, Cache


class Sequential(CacheModule):
    def __init__(self, *modules, names=None, **named_modules):
        assert (len(modules) > 0) ^ (
            len(named_modules) > 0
        ), "Either unnamed or named modules should be provided, but not both"
        super().__init__()

        self._sequence = nn.ModuleDict(modules)

    def forward(self, x, cache: Cache = None, **kwargs):
        for i, module in enumerate(self._sequence.values()):
            if isinstance(module, CacheModule):
                x = module(x, cache=cache[i], **kwargs)
            else:
                x = module(x)
        return x

    def __getitem__(self, key):
        return self._sequence[key]

    def __getattr__(self, key):
        if key in self._sequence:
            self._sequence[key]
        return super().__getattr__(key)


class NamedSequential(CacheModule):

    def forward(self, x, cache: Cache = None, **kwargs):
        for i, (name, module) in enumerate(self._sequence):
            if isinstance(module, CacheModule):
                x = module(x, cache=cache[name], **kwargs)
            else:
                x = module(x)
        return x

    # def __getattr__(self, name: str) -> Any:
    #     return super().__getattr__(name)
