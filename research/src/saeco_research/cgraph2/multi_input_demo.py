"""Demo: multi-input graphs via a *typed* ``DictBatch`` + ``field()`` -- declare a
batch contract, read its fields with no string accesses, seed one struct.

This is the primitive under docs/DESIGN-multi-input.md (a true ``total_loss``
handle, transcoders ``y_norm != x_norm``). Purely additive: the single-tensor
``Input`` path is unchanged; a graph either reads ``Input`` as the value or reads
``field(...)`` off a struct.

Three things shown:
1. **Typed fields.** ``Batch.input`` reads as ``torch.Tensor`` to the type checker
   (the metaclass makes the class-level field a typed ``FieldRef``); ``field(
   Batch.input)`` builds the reading node. No string keys.
2. **Transcoder shape.** ``x`` and ``y`` are *distinct* runtime inputs.
3. **Override escape hatch.** A field declared as a slot whose default reads the
   batch (``arg(default=field(Batch.input))``) can be bound to a plain value at
   construction -- so you aren't forced onto the batch input.
"""

import torch

import saeco.core as cl
from saeco_research.cgraph2 import DictBatch, GraphSpec, arg, field, op
from saeco_research.cgraph2.spec import Arg


def _l2(a, b):
    return ((a - b) ** 2).mean()


class Batch(DictBatch):
    input: torch.Tensor
    target: torch.Tensor
    weight: float


class TranscoderLoss(GraphSpec):
    """A weighted recon loss reading three distinct fields of the batch: the
    source ``input``, a separate ``target`` (so ``y != x``), and a scalar
    ``weight``. The "model" is a stand-in ``tanh``."""

    x = field(Batch.input)
    y = field(Batch.target)  # distinct from x -- the transcoder case
    w = field(Batch.weight)
    recon = op(torch.tanh)(x)
    output = w * op(_l2)(recon, y)


class OverridableLoss(GraphSpec):
    """Same loss, but ``x`` is an overridable slot defaulting to ``Batch.input`` --
    bind it to bypass the batch and feed a plain input instead."""

    x: Arg = arg(default=field(Batch.input))
    y = field(Batch.target)
    w = field(Batch.weight)
    recon = op(torch.tanh)(x)
    output = w * op(_l2)(recon, y)


def main() -> None:
    g = TranscoderLoss()
    # typed construction: pyright checks the field types + that none are missing
    batch = Batch(
        input=torch.randn(8, 4),
        target=torch.randn(8, 4),
        weight=2.0,
    )
    out = g.forward(batch, cache=cl.Cache())

    # 1. it computes weight * mse(tanh(input), target) over the typed batch
    expect = batch.weight * _l2(torch.tanh(batch.input), batch.target)
    assert torch.allclose(out, expect), (out, expect)

    # 2. x and y are genuinely distinct runtime inputs (the transcoder point)
    assert g.x is not g.y
    # reading the same field twice fuses to one node
    assert field(Batch.input) is field(Batch.input)

    # 3. a missing declared field fails loud (fail-at-the-boundary, principle 4).
    #    pyright flags the omitted `target` here too (typed construction) -- the
    #    ignore is deliberate: this is the negative test for the *runtime* guard.
    incomplete = Batch(input=batch.input, weight=2.0)  # type: ignore[call-arg]
    assert incomplete.missing_fields() == ["target"]
    try:
        g.forward(incomplete, cache=cl.Cache())
    except (KeyError, AttributeError):
        pass
    else:  # pragma: no cover
        raise AssertionError("missing field should raise")

    # 4. override: bind x to a fixed tensor so it bypasses the batch; y/w still
    #    come from the batch -- you aren't forced to route x through it.
    raw_x = torch.randn(8, 4)
    # binding a concrete value to a slot is runtime-handled (wrapped in a source
    # node); pyright sees the slot's declared `Arg` type -- expected friction.
    go = OverridableLoss(x=raw_x)  # type: ignore[arg-type]
    out_o = go.forward(batch, cache=cl.Cache())
    assert torch.allclose(
        out_o, batch.weight * _l2(torch.tanh(raw_x), batch.target)
    )

    print("typed multi_input_demo OK -- fields fuse into one pass:")
    print(g.render())


if __name__ == "__main__":
    main()
