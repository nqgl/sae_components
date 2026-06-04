"""Task #4 design spike: the total loss as one fused graph.

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph.megaloss_spike

Tests the load-bearing claim behind the "megagraph" plan: when main + aux +
penalty + per-loss math + the weighted sum are one DAG, the computation shared
between the main and aux models (here: the encoder) is done ONCE — replacing the
cross-call `ReuseForward` dedup with intra-graph identity dedup.

The catch this spike makes concrete: dedup is by NODE IDENTITY. So it matters
*how* you fuse:
  * naive  — main and aux built as separate sub-Graphs, each embedding its own
             encoder node, then nested as opaque nodes => encoder runs TWICE.
  * fused  — one GraphDef, a single shared `h = g(enc, x)` ref feeding both
             branches => encoder runs ONCE.
"Build duplicated then fuse" only dedups if fusion flattens + CSEs; sharing refs
up front gets there directly. This spike does the shared-ref version.

It also shows the bonus simplifications: the penalty is a *value node* on an edge
(no `cache.search`), and the aux branch's detach is an explicit `.detach()` edge.
"""

# ruff: noqa: N801
# (namespace classes intentionally read as values -- see cgraph/TASKS.md #6)
from __future__ import annotations

import torch

import saeco.core as cl
from saeco_research.cgraph import GraphDef, Op
from saeco_research.cgraph.demo import CountLinear

MSE = Op.create(lambda a, b: ((a - b) ** 2).mean(), "MSE")
L1 = Op.create(lambda a: a.abs().mean(0).sum(), "L1")
RELU = Op.create(torch.relu)


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def make_parts(d_in=8, d_dict=16):
    enc = CountLinear(d_in, d_dict)  # SHARED between main and aux
    dec_main = CountLinear(d_dict, d_in)
    dec_aux = CountLinear(d_dict, d_in)  # aux has its own decoder (gated-style)
    return enc, dec_main, dec_aux


# --------------------------------------------------------------------------- #
# naive: main and aux are separate opaque sub-graphs, each with its own encoder
# node. Nesting them does NOT fuse -> encoder runs once per sub-graph.
# --------------------------------------------------------------------------- #
def build_naive_total(enc, dec_main, dec_aux, w_aux=0.5):
    def model_graph(dec):
        g = GraphDef()
        x = g.input
        recon = g(dec, g(RELU, g(enc, x)))
        return g.build(recon)

    main_g, aux_g = model_graph(dec_main), model_graph(dec_aux)
    g = GraphDef()
    x = g.input
    main = g(main_g, x)  # opaque node: encoder hidden inside
    aux = g(aux_g, x)  # opaque node: its own encoder inside
    total = g(MSE, main, x) + g(MSE, aux, x) * w_aux
    return g.build(total)


# --------------------------------------------------------------------------- #
# fused: one GraphDef, a single shared encoder node `h`. Main, aux, and the
# penalty all reference it -> encoder runs once.
# --------------------------------------------------------------------------- #
def build_fused_total(enc, dec_main, dec_aux, w_aux=0.5, l1c=1e-2):
    g = GraphDef()

    @g.namespace
    class loss_graph:
        x = g.input
        h = g(enc, x)  # shared encoder, ONE node
        acts = g(RELU, h)
        main = g(dec_main, acts)  # main reconstruction
        aux = g(dec_aux, acts.detach())  # aux shares features (detached, gated-style)
        penval = g(L1, acts)  # penalty as a VALUE node on an edge
        l_main = g(MSE, main, x)
        l_aux = g(MSE, aux, x)
        total = l_main + l_aux * w_aux + penval * l1c
        g.output = total

    return loss_graph


def main():
    torch.manual_seed(0)
    x = torch.randn(32, 8)

    print("=== naive (opaque sub-graph nesting) ===")
    enc, dm, da = make_parts()
    naive = build_naive_total(enc, dm, da)
    enc.calls = 0
    naive(x, cache=cl.Cache())
    print(f"  encoder ran {enc.calls}x")
    ok("opaque nesting does NOT fuse (encoder runs twice)", enc.calls == 2)

    print("\n=== fused (one graph, shared encoder ref) ===")
    enc, dm, da = make_parts()
    fused = build_fused_total(enc, dm, da)
    print(fused)
    enc.calls = 0
    total = fused(x, cache=cl.Cache())
    print(f"  encoder ran {enc.calls}x; total loss = {total.item():.4f}")
    ok("fusion dedups (encoder runs once across main + aux + penalty)", enc.calls == 1)
    ok("total loss is a scalar", total.dim() == 0)

    print("\n=== it trains (one backward through the whole loss graph) ===")
    opt = torch.optim.Adam(
        [*enc.parameters(), *dm.parameters(), *da.parameters()], lr=1e-2
    )
    history = []
    for _ in range(150):
        loss = fused(x, cache=cl.Cache())
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    ok("loss decreased", history[-1] < history[0])
    print(f"  loss: {history[0]:.4f} -> {history[-1]:.4f}")

    # the aux decoder learns, but its grad does NOT flow back through the
    # detached encoder edge -- explicit, not implicit.
    loss = fused(x, cache=cl.Cache())
    enc.zero_grad()
    da.zero_grad()
    loss.backward()
    ok("aux decoder gets gradient", da.lin.weight.grad.abs().sum() > 0)
    print("\nall checks passed.")


if __name__ == "__main__":
    main()
