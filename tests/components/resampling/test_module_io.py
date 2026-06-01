"""Tests for `get_modules_io` / `ModuleIO` capture+shape behavior.

These exercise the forward-hook capture path with tiny `nn.Module`s and no
GPU/real-data infra, pinning the documented shape of `ModuleIO.inputs`:
`dict[nn.Module, list[tuple[Tensor, ...]]]` (one tuple of positional args per
forward call), which the resampler consumes via `inputs[module][call][arg_pos]`.
"""

import torch
import torch.nn as nn

from saeco.components.resampling.anthropic_resampling import (
    ModuleIO,
    get_modules_io,
)


class Inner(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


class Outer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.inner = Inner(d)

    def forward(self, x):
        return self.inner(x)


def test_get_modules_io_captures_input_args_tuple():
    d = 4
    model = Outer(d)
    data = torch.randn(3, d)

    res = get_modules_io(model, modules=[model.inner], data=data)

    assert isinstance(res, ModuleIO)
    # One forward call -> one captured entry per hooked module.
    captured = res.inputs[model.inner]
    assert len(captured) == 1
    # Each entry is the positional-args tuple, so [call][arg_pos] reaches a Tensor.
    args = captured[0]
    assert isinstance(args, tuple)
    assert len(args) == 1
    assert torch.equal(args[0], data.float())


def test_get_modules_io_model_out_and_outputs():
    d = 4
    model = Outer(d)
    data = torch.randn(2, d)

    res = get_modules_io(model, modules=[model.inner], data=data)

    # model_out is the model's forward result; here Outer just forwards Inner.
    expected = model(data.float())
    assert torch.equal(res.model_out, expected)

    # outputs captures the hooked module's forward return (here a single Tensor).
    outs = res.outputs[model.inner]
    assert len(outs) == 1
    assert torch.equal(outs[0], expected)


def test_get_modules_io_multiple_calls_append():
    d = 3

    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner(d)

        def forward(self, x):
            return self.inner(x) + self.inner(x)

    model = Twice()
    data = torch.randn(2, d)

    res = get_modules_io(model, modules=[model.inner], data=data)

    # The inner module is invoked twice per forward, so two entries accumulate.
    captured = res.inputs[model.inner]
    assert len(captured) == 2
    for args in captured:
        assert isinstance(args, tuple)
        assert torch.equal(args[0], data.float())


def test_get_modules_io_removes_hooks():
    d = 3
    model = Outer(d)
    data = torch.randn(2, d)

    get_modules_io(model, modules=[model.inner], data=data)

    # Hooks must be removed after capture so later forward passes are unaffected.
    assert len(model.inner._forward_hooks) == 0
