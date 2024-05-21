from typing import Any
import torch.nn as nn
from sae_components.core.cache import Cache
from sae_components.core.cache_layer import Module


class Seq(Module):
    def __init__(self, *modules, names=None, **named_modules):
        assert (len(modules) > 0) ^ (
            len(named_modules) > 0
        ), "Either unnamed or named modules should be provided, but not both"
        super().__init__()

        if len(modules) > 0:
            d = {str(i): module for i, module in enumerate(modules)}
        else:
            d = named_modules
        self._sequence = nn.ModuleDict(d)

    def forward(self, x, *, cache: Cache = None, **kwargs):
        for i, module in enumerate(self._sequence.values()):
            if isinstance(module, Module):
                x = module(x, cache=cache[i], **kwargs)
            else:
                x = module(x)
        return x

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self._sequence.values())[key]
        return self._sequence[key]

    def __getattr__(self, key):
        print(key)
        if key in super().__getattr__("_sequence"):
            return super().__getattr__("_sequence")[key]
        return super().__getattr__(key)

    # @property
    # def in_features(self):
    #     return self[0].in_features

    # @property
    # def out_features(self):
    #     return self[-1].out_features


class NamedSequential(Module):
    def forward(self, x, *, cache: Cache = None, **kwargs):
        for i, (name, module) in enumerate(self._sequence):
            if isinstance(module, Module):
                x = module(x, cache=cache[name], **kwargs)
            else:
                x = module(x)
        return x

    # def __getattr__(self, name: str) -> Any:
    #     return super().__getattr__(name)
