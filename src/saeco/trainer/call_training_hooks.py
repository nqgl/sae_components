from collections.abc import Callable
from typing import Any

import torch.nn as nn

from saeco.components.type_acc_methods import (
    PostBackwardHook,
    PostForwardHook,
    PostStepHook,
    PreForwardHook,
    typeacc_method,
)


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


def do_decorated_post_backward(module: nn.Module, cache=None):
    d: dict[nn.Module, dict[str, Callable[[], Any]]] = {}

    def appl_fn(m):
        fields = PostBackwardHook.get_fields(m)
        if fields:
            d[m] = {name: getattr(m, name) for name in fields}

    module.apply(appl_fn)
    for m, hooks in d.items():
        for name, hook in hooks.items():
            hook()


def call_decorated_training_hook(
    module: nn.Module, hook_type: type[typeacc_method], cache=None
):
    d: dict[nn.Module, dict[str, Callable[[], Any]]] = {}

    def appl_fn(m):
        fields = hook_type.get_fields(m)
        if fields:
            d[m] = {name: getattr(m, name) for name in fields}

    module.apply(appl_fn)
    for m, hooks in d.items():
        for name, hook in hooks.items():
            hook()


def do_post_backward(module: nn.Module, cache=None):
    find_and_call_attr_on_modules(module, "post_backward_hook")
    if cache is not None:
        find_and_call_attr_on_modules(
            module, "post_backward_hook_with_cache", cache=cache
        )
    call_decorated_training_hook(module, PostBackwardHook, cache=cache)


def do_post_step(module: nn.Module, cache=None):
    find_and_call_attr_on_modules(module, "post_step_hook")
    if cache is not None:
        find_and_call_attr_on_modules(module, "post_step_hook_with_cache", cache=cache)
    call_decorated_training_hook(module, PostStepHook, cache=cache)


def do_pre_forward(module: nn.Module, cache=None):
    find_and_call_attr_on_modules(module, "pre_forward_hook")
    if cache is not None:
        find_and_call_attr_on_modules(
            module, "pre_forward_hook_with_cache", cache=cache
        )
    call_decorated_training_hook(module, PreForwardHook, cache=cache)


def do_post_forward(module: nn.Module, cache=None):
    find_and_call_attr_on_modules(module, "post_forward_hook")
    if cache is not None:
        find_and_call_attr_on_modules(
            module, "post_forward_hook_with_cache", cache=cache
        )
    call_decorated_training_hook(module, PostForwardHook, cache=cache)
