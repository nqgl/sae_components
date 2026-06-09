"""cgraph v2 demo — relational graphs, global Input, interned fusion (task #7).

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph2.demo

Shows the v2 wins over v1:
  * no mutable builder, no `g.input()` -- `Input` is global; `x` defaults to it;
  * fusion is structural/global -- identical structure is the SAME interned node,
    so composing handles (or two specs) shares computation with no inline step;
  * `produce(handle)` runs only that handle's cone;
  * consing keys on producer/function identity, never a label.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import saeco.core as cl
from saeco_research.cgraph2 import (
    Arg,
    GraphSpec,
    Input,
    Slot,
    ValueRef,
    arg,
    op,
    op_slot,
)


class CountLinear(cl.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.calls = 0

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        self.calls += 1
        return self.lin(x)


class SAESpec(GraphSpec):
    encoder: Slot = op_slot()
    decoder: Slot = op_slot()
    x: Arg = arg(default=Input)  # defaults to the global Input
    acts = encoder(x)
    recon = decoder(acts)
    output = recon


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def run(graph, x):
    enc.calls = dec.calls = 0
    graph(x, cache=cl.Cache())
    return enc.calls, dec.calls


enc = CountLinear(8, 16)
dec = CountLinear(16, 8)


def main():
    torch.manual_seed(0)
    x_in = torch.randn(4, 8)

    sae = SAESpec(encoder=enc, decoder=dec)  # no x= needed

    ok("no g.input(): x defaults to the global Input", sae.x is Input)
    ok("acts/recon are handles", isinstance(sae.acts, ValueRef))

    # produce(handle) runs only that handle's cone
    ok("produce(acts) runs encoder only", run(sae.produce(sae.acts), x_in) == (1, 0))
    ok("produce(recon) runs enc + dec", run(sae.produce(sae.recon), x_in) == (1, 1))

    # structural fusion: recon-loss + penalty on acts -> encoder ONCE, just by
    # referencing the handles (no builder, no inline)
    recon_loss = op(lambda a, b: ((a - b) ** 2).mean())(Input, sae.recon)
    total = recon_loss + op(lambda a: a.abs().mean())(sae.acts) * 0.1
    enc_once = run(total.build(), x_in)[0] == 1
    ok("recon-loss + penalty(acts): encoder runs ONCE (interned)", enc_once)

    # two specs with the SAME producers are the SAME interned nodes (global fusion)
    sae2 = SAESpec(encoder=enc, decoder=dec)
    ok("two specs, same producers -> same interned acts node", sae.acts is sae2.acts)

    # different producers -> different nodes (the dedup tension, correct)
    sae3 = SAESpec(encoder=CountLinear(8, 16), decoder=dec)
    ok("different encoder -> different acts node", sae.acts is not sae3.acts)

    # consing keys on function identity, not labels
    same = op(torch.relu)(sae.acts) is op(torch.relu)(sae.acts)
    ok("two op(torch.relu) fuse (function id)", same)

    # the v1 `Nest` case, as a v2 spec: value slots default to the SAE's handles,
    # which share the encoder -> referencing both computes the encoder ONCE.
    class NestSpec(GraphSpec):
        recon: Arg = arg(default=sae.recon)
        acts: Arg = arg(default=sae.acts)
        output = op(torch.sum)(recon) + op(torch.sum)(acts)

    nest = NestSpec()
    nest_runs = run(nest.produce(nest.output), x_in)
    ok("Nest: encoder once, decoder once (shared handles)", nest_runs == (1, 1))

    # value slots can default to a param/tensor (bound as a source value node)
    class Biased(GraphSpec):
        bias: Arg = arg(default=nn.Parameter(torch.ones(8)))
        output = Input + bias

    out = Biased().produce(Biased().output)(x_in, cache=cl.Cache())
    ok("arg(default=param) binds a param as a source value", out.shape == x_in.shape)

    print("\nall checks passed.")


if __name__ == "__main__":
    main()
