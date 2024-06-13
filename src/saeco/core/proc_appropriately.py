from saeco.core.cache import Cache
from saeco.core.module import Module

import torch
import torch.nn as nn


CE_STR = "CacheExceptionLocationInfo: Except inside of subcache path: "


def proc_appropriately(module, name, x, cache: Cache, **kwargs):
    try:
        if isinstance(module, Module):
            val = module(x, cache=cache[name], **kwargs)
        elif isinstance(module, nn.Parameter):  # | torch.Tensor?
            return module
        else:
            val = module(x)
        cache._write(name, val)
    except Exception as e:
        if not any([CE_STR in a for a in e.args]):
            e.args = e.args + (f"{CE_STR}{cache._name}.{name}",)

        raise e
    return val
