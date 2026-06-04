"""Validation / readability demo for the cgraph prototype.

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph.demo

Two cases that today need `cl.ReuseForward` (gated encoder) or a shared-cache
write-back (affine normalizer). For each we:
  * build the computation as a DAG,
  * check it numerically against a plain-torch reference, and
  * prove the shared node runs exactly once (a call counter) -- i.e. the DAG
    gets for free what ReuseForward does by memoizing on (id(module), id(x)).
"""

# ruff: noqa: N801
# (namespace classes intentionally read as values -- @g.namespace returns the
# built Graph; see cgraph/TASKS.md #6 for the type-visible slot-model successor)
from __future__ import annotations

import torch
import torch.nn as nn
from attrs import define

import saeco.core as cl
from saeco_research.cgraph import GraphDef, Op, Ref


class CountLinear(cl.Module):
    """A bias-free linear that counts how many times it actually runs."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.calls = 0

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        self.calls += 1
        return self.lin(x)


def thresh() -> Op:
    return Op.create(lambda t: (t > 0).float(), "Thresh")


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


# --------------------------------------------------------------------------- #
# Case 1: gated encoder. The encoder output `h` feeds BOTH the gate and the
# magnitude branch. Today that shared `init.encoder` is wrapped in ReuseForward
# in both `enc_mag` and `enc_gate` (components/gated.py:82,94). Here it is one
# node with two out-edges.
# --------------------------------------------------------------------------- #
def gated_demo():
    print("\n=== gated encoder ===")
    torch.manual_seed(0)
    d_in, d_dict = 8, 16

    enc = CountLinear(d_in, d_dict)  # the shared node
    dec = nn.Linear(d_dict, d_in, bias=False)
    log_r = nn.Parameter(torch.randn(d_dict))
    mag_bias = nn.Parameter(torch.randn(d_dict))
    gate_bias = nn.Parameter(torch.randn(d_dict))

    g = GraphDef()

    @g.namespace
    class model:  # the decorated class becomes the built Graph
        x = g.input
        h = g(enc, x)  # encoder, computed once...
        expr = g(Op.create(torch.exp), g(log_r))  # label auto-derives to "exp"
        mag = g(nn.ReLU(), h * expr + g(mag_bias))  # ...used in mag...
        gate = g(thresh(), h + g(gate_bias))  # ...and in gate
        acts = gate * mag
        out = g(dec, acts)
        g.output = out

    print(model)

    x = torch.randn(4, d_in)

    # plain-torch reference (touches enc.lin directly, so the counter stays clean)
    h = enc.lin(x)
    ref = dec(thresh()(h + gate_bias) * torch.relu(h * torch.exp(log_r) + mag_bias))

    enc.calls = 0
    cache = cl.Cache()
    out = model(x, cache=cache)

    ok("matches reference", torch.allclose(out, ref, atol=1e-6))
    ok("shared encoder ran exactly once", enc.calls == 1)
    print("cache._unwatched_writes", cache._unwatched_writes)


# --------------------------------------------------------------------------- #
# Case 2: affine normalizer, kept contained. `apply` emits the normalize
# subgraph and returns a bundle of refs that owns `invert`; scale & shift are
# each consumed twice (normalize + invert) but computed once. Replaces the
# `cache.scale = ...` write-back + ReuseForward in NormalizedIO / AffineNormalizer
# (trainer/normalizers/normalizer.py:60-71,119-130).
# --------------------------------------------------------------------------- #
class MeanShift(cl.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        self.calls += 1
        return x.mean(dim=-1, keepdim=True)


class StdScale(cl.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.calls = 0

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        self.calls += 1
        return x.std(dim=-1, keepdim=True, unbiased=False) + self.eps


@define
class AffineApplied:
    """The refs produced by AffineNormalizer.apply; owns the inverse."""

    normed: Ref
    scale: Ref
    shift: Ref

    def invert(self, y: Ref) -> Ref:
        return y * self.scale + self.shift  # reuses the SAME scale/shift refs


@define(eq=False)
class AffineNormalizer:
    shift: cl.Module
    scale: cl.Module

    def apply(self, g: GraphDef, x: Ref) -> AffineApplied:
        shift = g(self.shift, x)
        x_s = x - shift
        scale = g(self.scale, x_s)
        return AffineApplied(normed=x_s / scale, scale=scale, shift=shift)


def build_inner_sae(d_in: int, d_dict: int):
    """A tiny SAE built as its own graph -- nested as a node in the outer graph."""
    enc = nn.Linear(d_in, d_dict)
    dec = nn.Linear(d_dict, d_in)
    g = GraphDef()

    @g.namespace
    class sae:
        x = g.input
        acts = g(nn.ReLU(), g(enc, x))
        recon = g(dec, acts)
        g.output = recon

    return sae


def normalizer_demo():
    print("\n=== affine normalizer (contained) wrapping a nested SAE ===")
    torch.manual_seed(0)
    d_in, d_dict = 8, 16

    shift_mod, scale_mod = MeanShift(), StdScale()
    norm = AffineNormalizer(shift_mod, scale_mod)
    inner = build_inner_sae(d_in, d_dict)

    g = GraphDef()

    @g.namespace
    class model:
        x = g.input
        nrm = norm.apply(g, x)  # emits shift / scale / normed
        y = g(inner, nrm.normed)  # nested SAE graph as a node
        out = nrm.invert(y)  # y*scale + shift, reusing scale & shift
        g.output = out

    print(model)

    x = torch.randn(4, d_in)

    # plain-torch reference for the affine wrap (inner graph reused as-is)
    shift = x.mean(dim=-1, keepdim=True)
    x_s = x - shift
    scale = x_s.std(dim=-1, keepdim=True, unbiased=False) + scale_mod.eps
    y = inner(x_s / scale, cache=cl.Cache())
    ref = y * scale + shift

    shift_mod.calls = scale_mod.calls = 0
    out = model(x, cache=cl.Cache())

    ok("matches reference", torch.allclose(out, ref, atol=1e-6))
    ok("shift computed once (used by normalize + invert)", shift_mod.calls == 1)
    ok("scale computed once (used by normalize + invert)", scale_mod.calls == 1)


if __name__ == "__main__":
    gated_demo()
    normalizer_demo()
    print("\nall checks passed.")
