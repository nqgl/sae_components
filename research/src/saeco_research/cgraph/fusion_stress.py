"""Deep-nesting fusion stress test (task #6).

Question: does ``A( B(C(x->P->d)->e) + B(F(x->P->d)->g) )`` run the shared
``x->P->d`` prefix ONCE? (C and F apply the same producer P to the same input,
then diverge; B wraps each; A sums.)

Answer: ONCE in all three realizations below:

  1. emit C/F into a SHARED builder      -> the shared x->P->d node conses -> ONCE
  2. reuse the SAME frozen subgraph obj  -> that subgraph node conses        -> ONCE
  3. DISTINCT frozen subgraph objects    -> applying a cgraph Graph now
     with identical internals               INLINES it (recursive cons)      -> ONCE

So fusion reaches through arbitrary depth: shared builders cons, and frozen Graph
objects inline on application, so their shared structure fuses either way.
"""

# ruff: noqa: N802, N803, N806
# (single-capital names P/C/F/B/A intentionally mirror the A(B(C..)) notation)
from __future__ import annotations

import torch
import torch.nn as nn

import saeco.core as cl
from saeco_research.cgraph.demo import CountLinear
from saeco_research.cgraph.slots import GraphDef, ValueRef


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def run(built, P) -> int:
    P.calls = 0
    built(torch.randn(4, 8), cache=cl.Cache())
    return P.calls


def main():
    torch.manual_seed(0)
    P = CountLinear(8, 8)  # the shared x->d producer; counts its runs
    Pe = nn.Linear(8, 8)
    Pg = nn.Linear(8, 8)

    # --- case 1: emit everything into one shared builder ---
    # C and F are just emitter calls; B wraps; A sums. P(x) conses across C and F.
    def C(g, x) -> ValueRef:
        return g(P, x)  # x -> d

    def F(g, x) -> ValueRef:
        return g(P, x)  # x -> d  (same producer, same input)

    def B(g, sub, tail) -> ValueRef:
        return g(tail, sub)  # d -> e / d -> g

    g = GraphDef()
    x = g.input()
    total = B(g, C(g, x), Pe) + B(g, F(g, x), Pg)  # A = the sum
    built = g.build(total)
    print(built)
    ok("(1) shared builder: x->P->d runs ONCE", run(built, P) == 1)

    # --- case 2: the SAME frozen subgraph object, reused ---
    def frozen_CF():
        gg = GraphDef()
        return gg.build(gg(P, gg.input()))

    cf = frozen_CF()  # ONE frozen graph object
    g = GraphDef()
    x = g.input()
    total = g(Pe, g(cf, x)) + g(Pg, g(cf, x))  # same cf node both sides
    ok("(2) same frozen subgraph reused: ONCE", run(g.build(total), P) == 1)

    # --- case 3: DISTINCT frozen subgraphs with identical internals ---
    # cgraph Graphs are our transparent structure, so applying one now INLINES it
    # (recursively, hash-consing). So even two separately-built graph objects with
    # the same internals fuse -- the opaque boundary dissolves by default.
    c_graph = frozen_CF()  # two separately-built graph objects...
    f_graph = frozen_CF()  # ...identical inside, but distinct objects
    g = GraphDef()
    x = g.input()
    total = g(Pe, g(c_graph, x)) + g(Pg, g(f_graph, x))
    once = run(g.build(total), P) == 1
    ok("(3) distinct frozen subgraphs now auto-inline -> ONCE", once)

    print("\nfusion reaches through nesting: shared builders cons, and frozen")
    print("Graph objects inline on application, so their shared structure fuses.")
    print("all checks passed.")


if __name__ == "__main__":
    main()
