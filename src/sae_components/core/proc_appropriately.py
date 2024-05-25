from sae_components.core.cache import Cache
from sae_components.core.module import Module

import torch
import torch.nn as nn


def proc_appropriately(module, name, x, cache: Cache, **kwargs):
    if isinstance(module, Module):
        val = module(x, cache=cache[name], **kwargs)
    elif isinstance(module, nn.Parameter):  # | torch.Tensor?
        return module
    else:
        val = module(x)
    cache._write(name, val)
    return val
