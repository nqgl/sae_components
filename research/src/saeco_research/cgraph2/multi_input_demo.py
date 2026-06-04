"""Demo: multi-input graphs via ``field()`` -- seed one structured batch at the
input and read named fields off it.

This is the *primitive* under docs/DESIGN-multi-input.md (the proposal that
unlocks a true ``total_loss`` handle and transcoders, ``y_norm != x_norm``). It is
purely additive: the single-tensor ``Input`` path is unchanged; a graph either
reads ``Input`` as the value (every existing graph) or reads ``field(...)`` off a
struct (here). Nothing in ``sae.py`` / the trainable is migrated onto fields yet --
that migration has open questions for Glen (see the design note); this just proves
the foundation is sound, fused, and device-clean.

A transcoder-shaped loss is the motivating case: the autoencoding *target* ``y``
is a distinct runtime input from the source ``x``, and the loss *weight* is a third
input -- all three are fields of one batch, and they all fuse into one pass.
"""

import torch

import saeco.core as cl
from saeco_research.cgraph2 import GraphSpec, field, op


def _l2(a, b):
    return ((a - b) ** 2).mean()


class TranscoderLoss(GraphSpec):
    """A weighted recon loss reading three distinct fields of the input batch:
    the source ``input``, a separate ``target`` (so ``y != x``), and a scalar
    ``weight``. The "model" here is a stand-in ``tanh``."""

    x = field("input")
    y = field("target")  # distinct from x -- the transcoder case
    w = field("weight")
    recon = op(torch.tanh)(x)
    output = w * op(_l2)(recon, y)


def main() -> None:
    g = TranscoderLoss()
    batch = {
        "input": torch.randn(8, 4),
        "target": torch.randn(8, 4),
        "weight": torch.tensor(2.0),
    }
    out = g.forward(batch, cache=cl.Cache())

    # 1. it computes weight * mse(tanh(x), y) over the structured input
    expect = batch["weight"] * _l2(torch.tanh(batch["input"]), batch["target"])
    assert torch.allclose(out, expect), (out, expect)

    # 2. x and y are genuinely *distinct* runtime inputs (the transcoder point)
    assert g.x is not g.y

    # 3. reading the same field twice fuses to one interned node (one extraction)
    assert field("input") is field("input")

    # 4. a missing field fails loud (fail-at-the-boundary, principle 4)
    incomplete = {"input": batch["input"], "weight": batch["weight"]}
    try:
        g.forward(incomplete, cache=cl.Cache())
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("missing field should raise")

    print("multi_input_demo OK -- three fields fuse into one pass:")
    print(g.render())


if __name__ == "__main__":
    main()
