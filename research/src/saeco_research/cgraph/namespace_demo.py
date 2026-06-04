"""Demo of the dataclass_transform class-namespace (task #6).

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph.namespace_demo

Same result as slots_demo, but the Loss is written as a class with annotated
slots. Proves: constructing the class binds slots + inlines into the builder, so
two losses sharing a model FUSE (model runs once); and pyright sees the
construction signature (checked separately in the typecheck probe).
"""

from __future__ import annotations

import torch

import saeco.core as cl
from saeco_research.cgraph.namespace import GraphSpec, arg, op_slot
from saeco_research.cgraph.slots import GraphDef, OpRef, ValueRef
from saeco_research.cgraph.slots_demo import build_sae


# A Loss template as a class: slots are annotated fields, the body wires them.
class Loss(GraphSpec):
    x: ValueRef = arg()  # value-input slot
    model: OpRef = op_slot()  # producer slot
    compare: OpRef = op_slot()  # producer slot
    y_pred = model(x)
    output = compare(x, y_pred)  # autoencoder: target == input


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    torch.manual_seed(0)
    sae, enc = build_sae(8, 16)

    g = GraphDef()
    x = g.input()
    model = g(sae)
    mse = g(lambda t, p: ((t - p) ** 2).mean(), label="MSE")
    mae = g(lambda t, p: (t - p).abs().mean(), label="MAE")

    # construct two losses sharing the SAME model + input; compose with +/*
    l_mse = Loss(x=x, model=model, compare=mse)
    l_mae = Loss(x=x, model=model, compare=mae)
    total = l_mse + l_mae * 0.5

    built = g.build(total)
    print(built)

    enc.calls = 0
    out = built(torch.randn(32, 8), cache=cl.Cache())
    ok("total loss is a scalar", out.dim() == 0)
    ok("model fused across two class-namespace losses -> ran once", enc.calls == 1)

    # --- slot defaults written the natural way: `compare` defaults to MSE via
    # a concrete `g(...)`, and is overridable at construction ---
    class WithDefault(GraphSpec):
        x: ValueRef = arg()
        model: OpRef = op_slot()
        compare: OpRef = g(lambda t, p: ((t - p) ** 2).mean())  # default MSE, via g()
        y_pred = model(x)
        output = compare(x, y_pred)

    sae2, _ = build_sae(8, 16)
    x_data = torch.randn(8, 8)

    g2 = GraphDef()
    default_out = g2.build(WithDefault(x=g2.input(), model=g2(sae2)).out)
    out_default = default_out(x_data, cache=cl.Cache())  # compare omitted -> MSE

    g3 = GraphDef()
    mae = g3(lambda t, p: (t - p).abs().mean())
    over = g3.build(WithDefault(x=g3.input(), model=g3(sae2), compare=mae).out)
    out_over = over(x_data, cache=cl.Cache())  # compare overridden -> MAE

    ok("concrete g(...) default used when slot omitted", out_default.dim() == 0)
    changed = not torch.allclose(out_default, out_over)
    ok("...and the override changes the graph (MAE != MSE)", changed)

    print("\nall checks passed.")


if __name__ == "__main__":
    main()
