"""Reusable "collection-role" graph components — reduce a *dynamic* collection of
named terms into one flat N-ary node, so the outer graph can stay mostly static.

The dynamic-count case (an unknown set of losses/metrics) can't be a class-body
``GraphSpec`` (the body runs before the collection is bound — you can't iterate a
slot at class-def). So it's built here imperatively over the *concrete* collection
and exposed as a clean reducer. Interning makes the result every bit as flat /
fused / deduped as a static graph: a single variadic node (no pairwise add-tree),
which torch reduces in one go and which you can use as a normal handle.
"""

from collections.abc import Iterable

import torch

from saeco_research.cgraph2.core import ValueRef, op


def _stack_sum(*terms):
    # one fused reduce over all terms at once (broadcast so heterogeneous shapes
    # still stack), not a pairwise tree.
    return torch.stack(torch.broadcast_tensors(*terms)).sum(0)


def reduce_sum(terms: Iterable[ValueRef]) -> ValueRef:
    """One flat N-ary node summing all ``terms`` at once (not a pairwise tree)."""
    return op(_stack_sum)(*terms)
