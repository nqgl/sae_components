from abc import ABCMeta
from typing import Any

import torch.nn as nn

import saeco.core as cl


class WrapsModule(cl.Module):
    wrapped: nn.Module

    def __init__(self, module):
        super().__init__()
        try:
            object.__getattribute__(self, "module")
            raise ValueError("module attribute already exists on self")
        except AttributeError:
            pass
        object.__setattr__(self, "module", module)
        self.wrapped = module

    def _get_name(self):
        return f"{self.__class__.__name__}[{self.wrapped._get_name()}]"

    def forward(self, *args, **kwargs):
        cache = kwargs.pop("cache", None)
        if cache:
            return cache(self).wrapped(*args, **kwargs)
        return self.wrapped(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("wrapped"), name)

    @classmethod
    def __instancecheck__(cls: ABCMeta, instance: Any) -> bool:
        return super().__instancecheck__(instance) or (
            isinstance(instance, WrapsModule) and isinstance(instance.wrapped, cls)
        )
