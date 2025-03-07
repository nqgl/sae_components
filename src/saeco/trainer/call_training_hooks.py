import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from saeco.components.sae_cache import SAECache
from torch import Tensor
from jaxtyping import Float

import saeco.core as cl


def find_and_call_attr_on_modules(module: nn.Module, attr: str, *args, **kwargs):
    """
    Calls a method on all modules in a module that have the method.
    Skips duplicates.
    """
    fns = {}

    def appl_fn(m):
        if hasattr(m, attr):
            fns[getattr(m, attr)] = m

    module.apply(appl_fn)
    return {m: fn(*args, **kwargs) for fn, m in fns.items()}


def do_post_backward(module: nn.Module, cache=None):
    find_and_call_attr_on_modules(module, "post_backward_hook")
    if cache is not None:
        find_and_call_attr_on_modules(
            module, "post_backward_hook_with_cache", cache=cache
        )


def do_post_step(module: nn.Module, cache=None):
    find_and_call_attr_on_modules(module, "post_step_hook")
    if cache is not None:
        find_and_call_attr_on_modules(module, "post_step_hook_with_cache", cache=cache)


def do_pre_forward(module: nn.Module, cache=None):
    find_and_call_attr_on_modules(module, "pre_forward_hook")
    if cache is not None:
        find_and_call_attr_on_modules(
            module, "pre_forward_hook_with_cache", cache=cache
        )


def do_post_forward(module: nn.Module, cache=None):
    find_and_call_attr_on_modules(module, "post_forward_hook")
    if cache is not None:
        find_and_call_attr_on_modules(
            module, "post_forward_hook_with_cache", cache=cache
        )
