"""Slot-model prototype demo (task #6).

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph.slots_demo

Proves the two things the slot model has to deliver:
 (1) a Loss is a HIGHER-ORDER graph (a `model` producer slot); binding the SAME
     model into two losses and fusing the total runs the model ONCE.
 (2) it's TYPE-VISIBLE today: the builder functions have typed slot parameters
     (`model: OpRef`), and `model(x)` returns a `ValueRef` -- pyright follows it
     with no metaclass. (This is the `model(x)` that was red under the
     decorator-returns-Graph namespace last turn.)
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

import saeco.core as cl
from saeco_research.cgraph.demo import CountLinear
from saeco_research.cgraph.slots import GraphDef, OpRef, ValueRef


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def build_sae(d_in: int, d_dict: int) -> tuple[cl.Module, CountLinear]:
    """An SAE as its own graph; returns it plus the (call-counting) encoder."""
    enc = CountLinear(d_in, d_dict)
    dec = nn.Linear(d_dict, d_in)
    g = GraphDef()
    x = g.input()
    acts = g(torch.relu, g(enc, x))
    return g.build(g(dec, acts)), enc


# A Loss is a graph template with a `model` producer slot and a `compare`
# producer slot -- ordinary typed parameters, so the slot signature is
# type-visible. `model(x)` is the call pyright couldn't follow last turn.
def build_loss(g: GraphDef, *, model: OpRef, compare: OpRef) -> ValueRef:
    x = g.input()
    y_pred = model(x)
    return compare(x, y_pred)  # autoencoder target == input


def main():
    torch.manual_seed(0)
    sae, enc = build_sae(8, 16)

    g = GraphDef()
    model = g(sae)  # producer (no inputs) -> OpRef; bound into BOTH losses below
    mse = g(lambda t, p: ((t - p) ** 2).mean(), label="MSE")
    mae = g(lambda t, p: (t - p).abs().mean(), label="MAE")

    # binding a slot == functools.partial (here: bind `compare`, leave `model` open)
    recon = partial(build_loss, compare=mse)
    l_mse = recon(g, model=model)
    l_mae = build_loss(g, model=model, compare=mae)
    total = l_mse + l_mae * 0.5

    built = g.build(total)
    print(built)

    x = torch.randn(32, 8)
    enc.calls = 0
    out = built(x, cache=cl.Cache())

    ok("total loss is a scalar", out.dim() == 0)
    ok("model bound into 2 losses ran ONCE (fused by hash-consing)", enc.calls == 1)
    # the model Graph is inlined+fused, so its encoder appears exactly once
    enc_nodes = [n for n in built._order if n.producer is enc]
    ok("model inlined+fused: exactly one encoder node", len(enc_nodes) == 1)

    # train a couple steps to confirm grads flow through the whole loss graph
    opt = torch.optim.Adam(built.parameters(), lr=1e-2)
    first = built(x, cache=cl.Cache()).item()
    for _ in range(50):
        loss = built(x, cache=cl.Cache())
        opt.zero_grad()
        loss.backward()
        opt.step()
    ok("loss decreased", built(x, cache=cl.Cache()).item() < first)

    print("\nall checks passed.")


if __name__ == "__main__":
    main()
