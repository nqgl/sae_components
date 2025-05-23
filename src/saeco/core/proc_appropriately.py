from saeco.core.cache import Cache
from saeco.core.module import Module
from saeco.misc.exception_location_hint import locate_cache_exception
import torch
import torch.nn as nn


# TODO unify/merge this and cache.__call__
# either merge, or make the behaviors identical to avoid inconsistent behavioir
def proc_appropriately(module, name, x, cache: Cache, **kwargs):
    try:
        if isinstance(module, Module) or (
            isinstance(module, torch._dynamo.eval_frame.OptimizedModule)
            and isinstance(module._orig_mod, Module)
        ):
            val = module(x, cache=cache[name], **kwargs)
        elif isinstance(module, nn.Parameter):  # | torch.Tensor?
            return module
        else:
            val = module(x)
        return cache._write(name, val)
    except Exception as e:
        raise locate_cache_exception(e, cache, name)
    return val
