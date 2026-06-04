"""Task #1: wire a graph-built SAE through the training machinery.

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph.trainable_spike

What this validates (the real integration risks, per PROD_READINESS.md):

1. **Seam.** A scratch `Architecture` overrides `make_trainable` to return a
   `MockTrainable(Trainable)` whose `self.model` is an *unnormalized* SAE graph
   wrapped by a *normalizing* graph (the agreed structure) — instead of
   `NormalizedIO`. `forward` flows through it.

2. **Cache integration.** A real `co.L1Penalty` placed as a graph node (deep
   inside a nested inner-SAE graph) writes `sparsity_penalty` into its subcache,
   and a real `co.SparsityPenaltyLoss` finds it via `cache.search(...)` — the
   graph executor preserves the nested subcache tree that `search` walks.

3. **The ReuseForward dependency we found.** `Loss.__init__` wraps the model in
   `ReuseForward` and `Loss.forward` re-runs it; across the per-loss re-runs in a
   step (sharing the root cache's `forward_reuse_dict`) the model — and the lone
   `sparsity_penalty` write — is deduped, so `search` returns exactly one. A
   graph model has no cross-call memo, so the naive per-loss re-run double-counts.
   The graph-native fix is **run once, let losses read the shared cache**, which
   removes BOTH the fan-out ReuseForward and this loss-layer one. `MockTrainable.loss`
   does that; we also show the naive path double-counting to make the point.

4. **It trains.** forward -> loss -> backward -> step reduces loss, and the
   penalty alone produces encoder gradients (so it's really wired through).
"""

# ruff: noqa: N801
# (namespace classes intentionally read as values -- see cgraph/TASKS.md #6)
from __future__ import annotations

import torch
import torch.nn as nn
from attrs import define

import saeco.core as cl
from saeco.components import L1Penalty, L2Loss, SparsityPenaltyLoss
from saeco.trainer.train_cache import TrainCache
from saeco.trainer.trainable import Trainable
from saeco_research.cgraph import GraphDef, Op
from saeco_research.cgraph.demo import (  # reuse the validated apply/invert pieces
    AffineNormalizer,
    MeanShift,
    StdScale,
)


class Linear(cl.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return self.lin(x)


def build_inner_sae(enc: Linear, dec: Linear) -> cl.Module:
    """Unnormalized SAE as its own graph, with a real L1Penalty pass-through node
    on the activations. This is what an `@model_prop` would return."""
    g = GraphDef()

    @g.namespace
    class sae:
        x = g.input
        acts = g(Op.create(torch.relu), g(enc, x))
        # L1Penalty is a PassThrough: writes sparsity_penalty, returns acts unchanged
        penalized = g(L1Penalty(), acts)
        recon = g(dec, penalized)
        g.output = recon

    return sae


def build_model(enc: Linear, dec: Linear, norm: AffineNormalizer) -> cl.Module:
    """Normalizing graph wrapping the inner SAE graph as a node — the analogue of
    today's `NormalizedIO(model)`, but as a DAG with no ReuseForward."""
    inner = build_inner_sae(enc, dec)
    g = GraphDef()

    @g.namespace
    class net:
        x = g.input
        nrm = norm.apply(g, x)  # normalize: emits shift/scale/normed
        y = g(inner, nrm.normed)  # inner SAE (penalty lives in here)
        out = nrm.invert(y)  # denormalize: reuses scale/shift
        g.output = out

    return net


class MockTrainable(Trainable):
    """Subclass of Trainable that deviates in two ways: (a) `self.model` is a
    graph (set directly, not via `normalizer.io_normalize`), and (b) `loss` runs
    the model ONCE and lets losses read the shared cache, instead of re-running
    the model per loss + relying on ReuseForward to dedup."""

    def __init__(self, model: cl.Module, losses: dict, coeffs: dict | None = None):
        cl.Module.__init__(self)  # skip Trainable.__init__ (SAE/normalizer-specific)
        self.model = model
        self._loss_objs = nn.ModuleDict(losses)
        self.coeffs = coeffs or {}

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return self.model(x, cache=cache)

    def loss(self, x, *, cache: cl.Cache, y=None, coeffs: dict | None = None):
        if y is None:
            y = x
        pred = self.model(x, cache=cache)  # ONE forward; populates the cache
        coeffs = {**self.coeffs, **(coeffs or {})}
        total = 0.0
        for key, loss_fn in self._loss_objs.items():
            term = loss_fn.loss(x, y, pred, cache)  # read shared cache; no re-run
            total = total + term * coeffs.get(key, 1.0)
        return total


@define(eq=False)
class ScratchArch:
    """Stand-in for an Architecture subclass. The real one would build these in
    `@model_prop`/`@loss_prop` and `make_trainable` would look identical."""

    d_in: int
    d_dict: int
    l1_coeff: float
    enc: Linear
    dec: Linear
    norm: AffineNormalizer

    @classmethod
    def create(cls, d_in: int, d_dict: int, l1_coeff: float = 1e-2) -> ScratchArch:
        return cls(
            d_in=d_in,
            d_dict=d_dict,
            l1_coeff=l1_coeff,
            enc=Linear(d_in, d_dict),
            dec=Linear(d_dict, d_in),
            norm=AffineNormalizer(MeanShift(), StdScale()),
        )

    def make_trainable(self) -> MockTrainable:
        model = build_model(self.enc, self.dec, self.norm)
        losses = {
            "L2_loss": L2Loss(model),
            "sparsity_penalty": SparsityPenaltyLoss(model),
        }
        return MockTrainable(model, losses, coeffs={"sparsity_penalty": self.l1_coeff})


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    torch.manual_seed(0)
    arch = ScratchArch.create(d_in=8, d_dict=16, l1_coeff=1e-2)
    tr = arch.make_trainable()

    print("=== model graph (normalizer wrapping a nested SAE) ===")
    print(tr.model)

    x = torch.randn(32, arch.d_in)

    # 2. cache integration: one forward, penalty is findable by search()
    print("\n=== cache integration ===")
    cache = TrainCache()
    pred = tr(x, cache=cache)
    pens = cache.search("sparsity_penalty")
    ok("forward produces a reconstruction", pred.shape == x.shape)
    ok("exactly one sparsity_penalty found via search()", len(pens) == 1)
    print(f"  sparsity_penalty = {pens[0].sparsity_penalty.item():.4f}")

    # 3. the ReuseForward dependency: naive per-loss re-run double-counts
    print("\n=== why production needs ReuseForward here (and the graph doesn't) ===")
    c = TrainCache()
    tr.model(x, cache=c["lossA"])  # re-run per loss, no cross-call memo...
    tr.model(x, cache=c["lossB"])
    n_naive = len(c.search("sparsity_penalty"))
    print(f"  naive per-loss re-run -> search finds {n_naive} penalties "
          f"(would trip SparsityPenaltyLoss's num_expected==1)")
    print("  run-once (MockTrainable.loss) -> 1, no ReuseForward needed")

    # 4. it trains, and the penalty is really wired to params
    print("\n=== train ===")
    opt = torch.optim.Adam(tr.parameters(), lr=1e-2)
    history = []
    for _ in range(150):
        cache = TrainCache()
        loss = tr.loss(x, cache=cache)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    ok("loss decreased", history[-1] < history[0])
    print(f"  loss: {history[0]:.4f} -> {history[-1]:.4f}")

    cache = TrainCache()
    pred = tr(x, cache=cache)
    sp = tr._loss_objs["sparsity_penalty"].loss(x, x, pred, cache)
    tr.zero_grad()
    sp.backward()
    grad = arch.enc.lin.weight.grad
    ok(
        "sparsity penalty alone produces encoder gradients",
        grad is not None and grad.abs().sum() > 0,
    )

    print("\nall checks passed.")


if __name__ == "__main__":
    main()
