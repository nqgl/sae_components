import inspect
from collections.abc import Callable
from typing import Any

import torch.nn as nn

from saeco.components.type_acc_methods import (
    PostBackwardHook,
    PostForwardHook,
    PostStepHook,
    PreForwardHook,
    saeco_hook,
    typeacc_method,
)
from saeco.components.wrap import WrapsModule

HookCall = tuple[Callable[..., Any], saeco_hook[Any, ...]]
HookCallsByTarget = dict[int, tuple[HookCall, ...]]


def iter_saeco_hook_targets(module: nn.Module):
    """Yield a logical module's raw module and transparent wrapper stack.

    ``WrapsModule.apply`` intentionally behaves like PyTorch's ``Module.apply``:
    it visits the wrapped module's children, then the wrapper itself as the
    logical module node. Saeco training hooks, however, may live on wrappers in
    that node's transparent wrapper stack, so hook discovery expands that stack
    explicitly.
    """
    wrappers = []
    target = module
    while isinstance(target, WrapsModule):
        wrappers.append(target)
        target = target.__wrapped__
    yield target
    yield from reversed(wrappers)


def _check_old_style_methods(module: nn.Module, *attrs: str):
    def appl_fn(m):
        for target in iter_saeco_hook_targets(m):
            for attr in attrs:
                try:
                    inspect.getattr_static(target, attr)
                except AttributeError:
                    continue
                else:
                    # If it has the old magic name but is decorated, that's fine.
                    if attr not in _old_style_hook_magic_names[attr].get_fields(
                        type(target)
                    ):
                        raise ValueError(
                            f"{type(target).__name__}.{attr} uses old name-based "
                            "hook discovery. Update it to the decorator API in "
                            "`saeco.components.type_acc_methods`."
                        )

    module.apply(appl_fn)


_old_style_hook_magic_names: dict[str, type[typeacc_method[Any, ...]]] = {
    "post_backward_hook": PostBackwardHook,
    "post_backward_hook_with_cache": PostBackwardHook,
    "post_step_hook": PostStepHook,
    "post_step_hook_with_cache": PostStepHook,
    "pre_forward_hook": PreForwardHook,
    "pre_forward_hook_with_cache": PreForwardHook,
    "post_forward_hook": PostForwardHook,
    "post_forward_hook_with_cache": PostForwardHook,
}


def call_decorated_training_hook(
    module: nn.Module, hook_type: type[saeco_hook[Any, ...]], cache=None
):
    d: HookCallsByTarget = {}

    def appl_fn(m):
        for target in iter_saeco_hook_targets(m):
            fields = hook_type.get_fields(type(target))
            if fields:
                d[id(target)] = tuple(
                    (
                        getattr(target, name),
                        inspect.getattr_static(target, name),
                    )
                    for name in fields
                )

    module.apply(appl_fn)
    for hooks in d.values():
        for hook, hook_descriptor in hooks:
            hook_descriptor.call(hook, cache=cache)


def do_post_backward(module: nn.Module, cache=None):
    _check_old_style_methods(
        module, "post_backward_hook", "post_backward_hook_with_cache"
    )
    call_decorated_training_hook(module, PostBackwardHook, cache=cache)


def do_post_step(module: nn.Module, cache=None):
    _check_old_style_methods(module, "post_step_hook", "post_step_hook_with_cache")
    call_decorated_training_hook(module, PostStepHook, cache=cache)


def do_pre_forward(module: nn.Module, cache=None):
    _check_old_style_methods(module, "pre_forward_hook", "pre_forward_hook_with_cache")
    call_decorated_training_hook(module, PreForwardHook, cache=cache)


def do_post_forward(module: nn.Module, cache=None):
    _check_old_style_methods(
        module, "post_forward_hook", "post_forward_hook_with_cache"
    )
    call_decorated_training_hook(module, PostForwardHook, cache=cache)
