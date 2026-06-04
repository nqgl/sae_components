"""Output-handles on a GraphSpec -- the cache replacement (assess).

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph.handles_demo

An `SAESpec` exposes its named intermediates (`acts`, `recon`) as grabbable
handles. You can:
  * build any handle -> runs only that handle's cone (`sae.produce(sae.acts)`
    skips the decoder);
  * splice a handle into another graph (the penalty), which fuses with the model.

This is the thing the cache does today (reach a value out of the middle of the
forward pass) -- but as a typed, first-class ref instead of a stringly-keyed
side channel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import saeco.core as cl
from saeco_research.cgraph.namespace import Arg, GraphSpec, Op, arg, op_slot
from saeco_research.cgraph.slots import GraphDef


class CountLinear(cl.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.calls = 0

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        self.calls += 1
        return self.lin(x)


# An SAE as a GraphSpec exposing `acts` and `recon` as handles.
class SAESpec(GraphSpec):
    x: Arg = arg()
    encoder: Op = op_slot()
    decoder: Op = op_slot()
    acts = encoder(x)  # handle: the activations (what a penalty wants)
    recon = decoder(acts)  # handle: the reconstruction
    output = recon


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    torch.manual_seed(0)
    enc = CountLinear(8, 16)
    dec = CountLinear(16, 8)
    x_in = torch.randn(4, 8)

    g = GraphDef()
    sae = SAESpec(x=g.input(), encoder=g(enc, label="enc"), decoder=g(dec, label="dec"))

    # grab handles by name (typed refs, not cache strings)
    grabbable = hasattr(sae, "acts") and hasattr(sae, "recon")
    ok("acts and recon are grabbable handles", grabbable)

    # build targeting `acts` -> only the encoder runs (decoder skipped)
    enc.calls = dec.calls = 0
    sae.produce(sae.acts)(x_in, cache=cl.Cache())
    ok("produce(acts) runs the encoder...", enc.calls == 1)
    ok("...and NOT the decoder", dec.calls == 0)

    # build targeting `recon` -> both run
    enc.calls = dec.calls = 0
    sae.produce(sae.recon)(x_in, cache=cl.Cache())
    ok("produce(recon) runs encoder + decoder", enc.calls == 1 and dec.calls == 1)

    # splice the `acts` handle into a penalty graph; it fuses with the recon path,
    # so building (recon-loss + penalty) runs the encoder ONCE (no cache needed)
    recon_loss = g(lambda a, b: ((a - b) ** 2).mean(), label="L2")(sae.x, sae.recon)
    penalty = g(lambda a: a.abs().mean(), label="L1")(sae.acts)
    total = recon_loss + penalty * 0.1
    enc.calls = 0
    total.build()(x_in, cache=cl.Cache())
    ok("penalty splices the acts handle; encoder runs once (fused)", enc.calls == 1)

    print("\nall checks passed.")
    _sae = sae
    enc.calls = dec.calls = 0

    # nesting it inside another graph
    class Nest(GraphSpec):
        x: Arg = arg()
        sae: Op = op_slot(default=_sae.produce(_sae.output))
        acts: Op = op_slot(default=_sae.produce(_sae.acts))
        sum_output = g(torch.sum, acts(x)) + g(torch.sum, sae(x))
        output = sum_output

    nest = Nest(x=g.input())

    nest.produce(nest.output)(x_in, cache=cl.Cache())

    ok("Nest.produce(Nest.output) runs the encoder...", enc.calls == 1)
    ok("...and also the decoder", dec.calls == 1)


if __name__ == "__main__":
    main()
