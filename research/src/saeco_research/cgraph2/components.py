"""Reusable "collection-role" graph components — reduce a *dynamic* collection of
named terms into one flat N-ary node, so the outer graph can stay mostly static.

The dynamic-count case (an unknown set of losses/metrics) can't be a class-body
``GraphSpec`` (the body runs before the collection is bound — you can't iterate a
slot at class-def). So it's built here imperatively over the *concrete* collection
and exposed as a clean reducer. Interning makes the result every bit as flat /
fused / deduped as a static graph: a single variadic node (no pairwise add-tree),
which torch reduces in one go and which you can use as a normal handle.
"""

from collections.abc import Callable, Iterable

import torch

from saeco_research.cgraph2.core import ValueRef, field, op


def _stack_sum(*terms):
    # one fused reduce over all terms at once (broadcast so heterogeneous shapes
    # still stack), not a pairwise tree.
    return torch.stack(torch.broadcast_tensors(*terms)).sum(0)


def reduce_sum(terms: Iterable[ValueRef]) -> ValueRef:
    """One flat N-ary node summing all ``terms`` at once (not a pairwise tree)."""
    return op(_stack_sum)(*terms)


def weighted_sum(
    terms: dict[str, ValueRef],
    weight: Callable[[str], str] = lambda k: f"w_{k}",
) -> ValueRef:
    """``Σ field(weight(k)) · term_k`` as one flat N-ary node — the total-loss reduce
    over a *dynamic* dict of named terms. ``weight(k)`` names the per-term weight
    field (default ``w_<k>``), seeded at the boundary each step (device-clean: the
    weights are graph inputs, not captured buffers — see D-04). The reduction is a
    single node, so it's flat and torch-friendly."""
    return reduce_sum(field(weight(k)) * term for k, term in terms.items())
