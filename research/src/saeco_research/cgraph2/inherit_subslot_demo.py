"""GraphSpec inheritance + sub-model slots (``sub_slot``) + the ``at`` move.

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph2.inherit_subslot_demo
"""

from typing import Any

import torch
import torch.nn as nn

import saeco.core as cl
from saeco_research.cgraph2 import (
    Arg,
    GraphSpec,
    Input,
    Slot,
    ValueRef,
    arg,
    at,
    op_slot,
    sub_slot,
)


def _mse(a, b):
    return ((a - b) ** 2).mean()


def _sumsq(a, b):
    return ((a - b) ** 2).sum()


def _l1(a):
    return a.abs().mean(0).sum()


class CountLinear(cl.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.calls = 0

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        self.calls += 1
        return self.lin(x)


# a model, as a GraphSpec
class SAESpec(GraphSpec):
    encoder: Slot = op_slot()
    decoder: Slot = op_slot()
    x: Arg = arg(default=Input)
    acts = encoder(x)
    recon = decoder(acts)
    output = recon


# a loss base that plugs a model in via a sub-model slot; concrete subclasses
# override the loss_fn default (exercising inheritance).
class ReconLoss(GraphSpec):
    model: Any = sub_slot()
    loss_fn: Slot = op_slot()
    pred = model(Input)  # plug the sub-model in, applied to this spec's input
    output = loss_fn(Input, pred.recon)


class L2Loss(ReconLoss):
    loss_fn: Slot = op_slot(default=_mse)


class SumSqLoss(ReconLoss):
    loss_fn: Slot = op_slot(default=_sumsq)


class SparsityLoss(GraphSpec):
    model: Any = sub_slot()
    penalty: Slot = op_slot(default=_l1)
    pred = model(Input)
    output = penalty(pred.acts)


# an abstract base: no `output` -> cannot be constructed
class AbstractBase(GraphSpec):
    thing: Slot = op_slot()


# a one-off spec that just exposes the sub-model's recon applied to an input,
# to compare the sub_slot path against the `at` primitive.
class JustRecon(GraphSpec):
    model: Any = sub_slot()
    x: Arg = arg()
    output = model(x).recon


# two model graphs plugged in, diffing their acts (Glen's ActDiff)
class ActDiff(GraphSpec):
    model_1: Any = sub_slot()
    model_2: Any = sub_slot()
    acts_1 = model_1(Input).acts
    acts_2 = model_2(Input).acts
    output = acts_1 - acts_2


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    torch.manual_seed(0)
    enc, dec = CountLinear(8, 16), CountLinear(16, 8)
    x = torch.randn(4, 8)
    sae = SAESpec(encoder=enc, decoder=dec)

    # --- sub-model slot: L2Loss plugs sae in; loss_fn defaults to _mse ---
    l2 = L2Loss(model=sae)
    ok("L2Loss.out is a handle", isinstance(l2.out, ValueRef))
    enc.calls = dec.calls = 0
    val = l2.out.build()(x, cache=cl.Cache())
    ok("L2 loss is a scalar", val.dim() == 0)
    ok("encoder ran once building the loss", enc.calls == 1)

    # --- inheritance: the overridden loss_fn actually changed the op ---
    mse_v = float(l2.out.build()(x, cache=cl.Cache()))
    sum_v = float(SumSqLoss(model=sae).out.build()(x, cache=cl.Cache()))
    ok("override loss_fn: _sumsq != _mse", abs(sum_v - mse_v) > 1e-3)

    # --- sub-model applied, reading .acts: encoder runs, decoder does NOT ---
    sp = SparsityLoss(model=sae)
    enc.calls = dec.calls = 0
    sp.out.build()(x, cache=cl.Cache())
    ok("sparsity reads .acts: enc ran, dec did not", enc.calls == 1 and dec.calls == 0)

    # --- the `at` primitive equals the sub_slot application (same interned node) ---
    src = arg(default=nn.Parameter(torch.randn(8)))  # a source node to rebind onto
    via_subslot = JustRecon(model=sae, x=src).out
    via_at = at(sae.recon, src)
    ok("at(recon, src) is model(src).recon (interned-equal)", via_subslot is via_at)

    # --- a bound graph is appliable too: sae(x).acts == at(sae.acts, x) ---
    ok("sae(src).acts is at(sae.acts, src)", sae(src).acts is at(sae.acts, src))

    # --- ActDiff: plug two model graphs in, diff their acts ---
    enc2 = CountLinear(8, 16)
    sae_b = SAESpec(encoder=enc2, decoder=dec)
    diff = ActDiff(model_1=sae, model_2=sae_b)
    enc.calls = enc2.calls = 0
    diff.out.build()(x, cache=cl.Cache())
    ok("ActDiff runs both models' encoders once", enc.calls == 1 and enc2.calls == 1)

    # --- abstract base (no output) cannot be constructed ---
    try:
        AbstractBase(thing=enc)
        constructed = True
    except TypeError:
        constructed = False
    ok("abstract base cannot be constructed", not constructed)

    print("\nall checks passed.")


if __name__ == "__main__":
    main()
