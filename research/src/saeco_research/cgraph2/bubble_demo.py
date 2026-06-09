"""Bubble-up sites: pull L0/L1/freqs out of a graph without the outside knowing
where they live (apply-style ``collect``), and defer a freq tracker's stateful
update onto the ``forward_values`` pass.

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph2.bubble_demo
"""

import torch
import torch.nn as nn

import saeco.core as cl
from saeco_research.cgraph2 import (
    GraphSpec,
    Input,
    Slot,
    op,
    op_slot,
)
from saeco_research.cgraph2.sae import FreqTracker, _MetricsSite, metered


class Lin(cl.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return self.lin(x)


# a model that splices a metrics site and a freq site onto its acts -- nothing
# downstream (the decoder) knows they're there; they bubble up on demand.
freqs = FreqTracker(16)


class SAESpec(GraphSpec):
    encoder: Slot = op_slot()
    decoder: Slot = op_slot()
    raw = op(torch.relu)(encoder(Input))
    acts = op(freqs)(metered(raw))  # metrics site then freq site, passthrough
    recon = decoder(acts)
    output = recon


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    torch.manual_seed(0)
    enc, dec = Lin(8, 16), Lin(16, 8)
    sae = SAESpec(encoder=enc, decoder=dec)
    g = sae  # the spec is itself the runnable graph
    x = torch.randn(32, 8)

    # bubble up the sites from the graph (the outside never reached "inside")
    metric_nodes = g.collect(_MetricsSite)
    freq_nodes = g.collect(FreqTracker)
    ok("collected one metrics site", len(metric_nodes) == 1)
    ok("collected one FreqTracker site", len(freq_nodes) == 1)

    # each site derives handles from its node's input (so they track rebinding)
    mnode = metric_nodes[0]
    metric_handles = mnode.producer.at(mnode.inputs[0])  # ActMetrics graph in context
    fnode = freq_nodes[0]
    ft = fnode.producer

    ok("freqs start at zero", float(freqs.freqs.sum()) == 0.0)

    # one fused pass: produce recon + L0/L1, and defer the freq update inline
    with g.forward_values(x, cache=cl.Cache()) as save:
        recon = save(sae.recon)
        l0 = save(metric_handles.L0)
        l1 = save(metric_handles.L1)
        # the freqs are still zero here -- update is deferred to block exit
        inside_zero = float(freqs.freqs.sum()) == 0.0
        ok("freqs still zero *inside* the block (deferred)", inside_zero)
        ft.update(ft.updated(fnode.inputs[0]))  # inline, runs after the pass

    ok("recon shape", recon.shape == x.shape)
    ok("L0 is a scalar", l0.dim() == 0)
    ok("L1 is a scalar", l1.dim() == 0)
    ok("freqs updated after the block", float(freqs.freqs.sum()) > 0.0)

    # second step: EMA accumulates (deferred update each pass)
    before = freqs.freqs.clone()
    with g.forward_values(torch.randn(32, 8), cache=cl.Cache()) as save:
        save(sae.recon)
        ft.update(ft.updated(fnode.inputs[0]))
    ok("freqs change again next step (EMA)", not torch.allclose(before, freqs.freqs))

    # the low-level primitive `save.after(handle, callback)` defers a plain callback
    # to block exit (the materialized handle value is passed in). `ft.update` itself
    # is a @deferred_execution training callback, so its idiom is the direct
    # `ft.update(ft.updated(acts))` above; `after` is for an arbitrary callback.
    freqs2 = FreqTracker(16)

    class SAE2(GraphSpec):
        encoder: Slot = op_slot()
        decoder: Slot = op_slot()
        acts = op(freqs2)(op(torch.relu)(encoder(Input)))
        recon = decoder(acts)
        output = recon

    s2 = SAE2(encoder=enc, decoder=dec)
    g2 = s2
    fn2 = g2.collect(FreqTracker)[0]

    @torch.no_grad()
    def write_freqs(v):
        freqs2.freqs.copy_(v)

    with g2.forward_values(torch.randn(32, 8), cache=cl.Cache()) as save:
        save(s2.recon)
        save.after(fn2.producer.updated(fn2.inputs[0]), write_freqs)
    ok("explicit save.after also updates freqs", float(freqs2.freqs.sum()) > 0.0)

    print("\nall checks passed.")


if __name__ == "__main__":
    main()
