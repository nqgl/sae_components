"""The dedup-vs-override tension (the nl1/nl2 question).

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph.dedup_test

Two ways to write "two nonlinearities", with opposite trade-offs:

  * DISTINCT instances -> nl1, nl2 are *independent* slots (overridable
    separately), but their identical defaults DON'T dedup: ReLU runs twice.
  * SHARED instance    -> nl1, nl2 are the *same* producer: deduped (ReLU once),
    but they can't be overridden independently (one identity).

This is the "larger problem" Glen flagged: hash-consing keys on producer
*identity*, so two distinct stateless-module instances look different even though
they're semantically equal. Mitigations: share the instance (below), or use a
function ref (`torch.relu`) which dedups by function identity, or (future) a
canonicalization pass for parameter-free producers. See PROD_READINESS.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import saeco.core as cl
from saeco_research.cgraph.namespace import Arg, GraphSpec, Op, arg, g
from saeco_research.cgraph.slots import GraphDef


class CountReLU(cl.Module):
    total = 0  # class-level: counts ReLU executions across all instances

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        type(self).total += 1
        return torch.relu(x)


class GDistinct(GraphSpec):
    x: Arg = arg()
    nl1: Op = g(CountReLU())  # two DISTINCT instances -> two independent slots
    nl2: Op = g(CountReLU())
    output = nl1(x) + nl2(x)


class GShared(GraphSpec):
    x: Arg = arg()
    relu = g(CountReLU())  # NOT annotated -> a shared producer, not a slot
    nl1: Op = relu
    nl2: Op = relu
    output = nl1(x) + nl2(x)


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def build_run(cls, **binds):
    gg = GraphDef()
    CountReLU.total = 0
    out = cls(x=gg.input(), **binds).out
    return gg.build(out)(torch.randn(4, 8), cache=cl.Cache()), CountReLU.total


def main():
    torch.manual_seed(0)

    # independent override (raw modules auto-wrap): the three are all different
    o0, calls = build_run(GDistinct)  # ReLU + ReLU
    o1, _ = build_run(GDistinct, nl2=nn.GELU())  # ReLU + GELU
    o2, _ = build_run(GDistinct, nl1=nn.GELU(), nl2=nn.GELU())  # GELU + GELU
    ok("G(x) != G(x, nl2=GELU)", not torch.allclose(o0, o1))
    ok("G(x, nl2=GELU) != G(x, nl1=GELU, nl2=GELU)", not torch.allclose(o1, o2))
    ok("G(x) != G(x, nl1=GELU, nl2=GELU)", not torch.allclose(o0, o2))

    # ...but the cost of independence: identical distinct defaults run twice
    ok("distinct defaults compute ReLU TWICE (redundant)", calls == 2)

    # sharing the instance dedups -> ReLU once (but the two are no longer
    # independently overridable: one identity)
    _, shared_calls = build_run(GShared)
    ok("shared instance dedups -> ReLU computed ONCE", shared_calls == 1)

    print("\nall checks passed.")


if __name__ == "__main__":
    main()
