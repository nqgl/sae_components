"""Task #4: automatic fusion of independently-authored subgraphs (hash-consing).

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph.fusion_spike

The point you raised: the encoder nodes in two subgraphs do the same thing (same
module, same input) but are distinct node *objects*, so identity-dedup alone
won't merge them. The builder now hash-conses: `g(producer, *inputs)` returns an
existing node when `(producer, input-nodes)` already exists. So two functions
written with NO knowledge of each other — `emit_main` and `emit_aux` below —
get the *same* encoder node back, and the redundancy is gone automatically.

This is the "assemble subgraphs without awareness of each other, redundancies
fuse away" property, realized by sharing a builder. (Merging already-frozen
separate Graphs would instead need an inline+CSE pass; sharing a builder gets it
for free, which is why the architecture should *emit into one builder* rather
than freeze subgraphs early.)
"""

from __future__ import annotations

import torch

import saeco.core as cl
from saeco_research.cgraph import GraphDef, Op
from saeco_research.cgraph.demo import CountLinear

RELU = Op.create(torch.relu)
MSE = Op.create(lambda a, b: ((a - b) ** 2).mean(), "MSE")


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


# Two model builders authored independently. Neither references the other; they
# only share the builder `g`, the input ref `x`, and the `enc` module instance.
def emit_main(g, x, enc, dec_main):
    h = g(enc, x)
    acts = g(RELU, h)
    return g(dec_main, acts)


def emit_aux(g, x, enc, dec_aux):
    h = g(enc, x)  # SAME enc + SAME x -> hash-consing returns main's `h` node
    acts = g(RELU, h)
    return g(dec_aux, acts.detach())  # aux detaches its features (gated-style)


def main():
    torch.manual_seed(0)
    enc = CountLinear(8, 16)  # shared encoder
    dec_main = CountLinear(16, 8)
    dec_aux = CountLinear(16, 8)

    g = GraphDef()
    x = g.input
    main = emit_main(g, x, enc, dec_main)  # built independently...
    aux = emit_aux(g, x, enc, dec_aux)  # ...and unaware of main
    total = g(MSE, main, x) + g(MSE, aux, x) * 0.5
    model = g.build(total)

    print("=== fused graph (note: ONE encoder node, ONE relu) ===")
    print(model)

    x_in = torch.randn(32, 8)
    enc.calls = 0
    out = model(x_in, cache=cl.Cache())
    print(f"\n  encoder ran {enc.calls}x")
    ok("independently-authored subgraphs auto-fused (encoder once)", enc.calls == 1)
    ok("output is a scalar loss", out.dim() == 0)

    # the encoder node has two consumers (main path + aux's detach), proving the
    # share is real structure, not luck.
    enc_nodes = [n for n in model._order if n.producer is enc]
    consumers = [n for n in model._order if any(i is enc_nodes[0] for i in n.inputs)]
    ok("exactly one encoder node in the graph", len(enc_nodes) == 1)
    print(f"  encoder feeds {len(consumers)} consumer(s): "
          f"{[model._names[c] for c in consumers]}")

    # opt-out: an impure producer can force its own node with fresh=True
    g2 = GraphDef()
    x2 = g2.input
    a = g2(enc, x2)
    b = g2(enc, x2, fresh=True)
    ok("fresh=True opts out of consing (distinct nodes)", a._node is not b._node)

    print("\nall checks passed.")


if __name__ == "__main__":
    main()
