"""#3: the gated SAE architecture, rewritten on the slot/graph model.

Parallels the saeco structure as closely as the prototype allows:

  * `SAEApplied`           ~ what an SAE produces (recon + the penalized acts).
  * `GraphLoss` / `GraphL2Loss` / `GraphSparsityLoss`  ~ `Loss` / `L2Loss` /
    `SparsityPenaltyLoss` -- each wraps a model and emits its loss *term* as a
    graph node (instead of re-running the model and reading the cache).
  * `GraphTrainable(Trainable)`  ~ `Trainable`, but `loss()` builds the total
    loss as ONE fused graph and runs it once.
  * `GatedGraph`           ~ the `Gated` architecture: the same `enc_mag`/
    `enc_gate`/`gated_model`/`model_aux` structure, with `encoder` shared by
    enc_mag+enc_gate and `enc_gate` shared by main+aux.

The payoff: the encoder and enc_gate -- shared across the main model, the aux
model, and all three losses -- are computed ONCE by hash-consing, replacing
every `ReuseForward` in the original gated arch (encoder-reuse AND the cross-loss
reuse). Leaf modules are plain Linears standing in for the real Initializer.
"""

from __future__ import annotations

from functools import reduce

import torch
import torch.nn as nn
from attrs import define

import saeco.core as cl
from saeco.trainer.trainable import Trainable
from saeco_research.cgraph.core import Op
from saeco_research.cgraph.slots import GraphDef, ValueRef


# --- leaf modules (stand in for init.encoder / init.decoder) ---
class CountLinear(cl.Module):
    total = 0  # class-level: counts encoder executions

    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        type(self).total += 1
        return self.lin(x)


class Linear(cl.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return self.lin(x)


def _thresh(t):
    return (t > 0).float()


def _mse(a, b):
    return ((a - b) ** 2).mean()


def _l1(a):
    return a.abs().mean(0).sum()


# --- what a model produces (parallels SAE: encode -> acts -> decode) ---
@define
class SAEApplied:
    recon: ValueRef
    acts: ValueRef  # the activations the sparsity penalty is applied to


# --- GraphLoss ~ Loss(module): wraps a model, emits its loss term as a node ---
@define
class GraphLoss:
    model: object  # an emitter with `.apply(g, x) -> SAEApplied`

    def term(self, g: GraphDef, x: ValueRef) -> ValueRef:
        return self.compute(g, x, self.model.apply(g, x))

    def compute(self, g, x, applied: SAEApplied) -> ValueRef:
        raise NotImplementedError


class GraphL2Loss(GraphLoss):  # ~ L2Loss
    def compute(self, g, x, applied):
        return g(_mse, label="L2")(x, applied.recon)


class GraphSparsityLoss(GraphLoss):  # ~ SparsityPenaltyLoss (penalty as a value edge)
    def compute(self, g, x, applied):
        return g(_l1, label="L1")(applied.acts)


# --- the two models (parallel @model_prop / @aux_model_prop) ---
@define
class _MainModel:  # ~ gated_model
    arch: GatedGraph

    def apply(self, g, x) -> SAEApplied:
        gate = g(_thresh)(self.arch.enc_gate(g, x))
        acts = gate * self.arch.enc_mag(g, x)
        return SAEApplied(recon=g(self.arch.dec_main, acts), acts=acts)


@define
class _AuxModel:  # ~ model_aux
    arch: GatedGraph

    def apply(self, g, x) -> SAEApplied:
        gate_acts = self.arch.enc_gate(g, x)  # SHARED with the main model's gate
        recon = g(self.arch.dec_aux, gate_acts.detach())  # detached decoder
        return SAEApplied(recon=recon, acts=gate_acts)


class GatedGraph(nn.Module):  # ~ Gated(Architecture[GatedConfig]); a container of parts
    def __init__(self, d_in: int, d_dict: int):
        super().__init__()
        self.encoder = CountLinear(d_in, d_dict)  # SHARED by enc_mag + enc_gate
        self.dec_main = Linear(d_dict, d_in)
        self.dec_aux = Linear(d_dict, d_in)
        self.gate_bias = nn.Parameter(torch.zeros(d_dict))
        self.mag_bias = nn.Parameter(torch.zeros(d_dict))
        self.log_r = nn.Parameter(torch.zeros(d_dict))

    # NB: use the function ref `torch.relu` (not `nn.ReLU()` instances) so the
    # nonlinearity dedups by identity -- enc_gate is emitted by main + aux + the
    # sparsity loss, and we want its relu computed once (see dedup_test.py).
    def enc_gate(self, g, x) -> ValueRef:  # ~ enc_gate (relu'd gate activations)
        h = g(self.encoder, x)
        return g(torch.relu, h + g(self.gate_bias))

    def enc_mag(self, g, x) -> ValueRef:  # ~ enc_mag
        h = g(self.encoder, x)  # SAME encoder -> conses with enc_gate's h
        exp_r = g(torch.exp)(g(self.log_r))
        return g(torch.relu, h * exp_r + g(self.mag_bias))

    @property
    def gated_model(self) -> _MainModel:
        return _MainModel(self)

    @property
    def model_aux(self) -> _AuxModel:
        return _AuxModel(self)

    @property
    def losses(self) -> dict[str, GraphLoss]:  # ~ the @loss_prop set
        return {
            "L2_loss": GraphL2Loss(self.gated_model),
            "L2_aux_loss": GraphL2Loss(self.model_aux),
            "sparsity_loss": GraphSparsityLoss(self.model_aux),
        }

    def make_trainable(self) -> GraphTrainable:
        return GraphTrainable(
            self, self.losses, coeffs={"L2_aux_loss": 1.0, "sparsity_loss": 3e-4}
        )


# --- GraphTrainable ~ Trainable, with the fused-graph loss calculation ---
class GraphTrainable(Trainable):
    def __init__(self, arch: GatedGraph, losses: dict, coeffs: dict):
        cl.Module.__init__(self)  # skip Trainable.__init__
        self.arch = arch  # registers the leaf modules / params
        self.graph_losses = losses
        self.coeffs = coeffs

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        g = GraphDef()
        recon = self.arch.gated_model.apply(g, g.input()).recon
        return g.build(recon)(x, cache=cache)

    def loss(self, x, *, cache: cl.Cache, y=None, coeffs=None):
        g = GraphDef()
        xin = g.input()
        coeffs = {**self.coeffs, **(coeffs or {})}
        # every loss emits into the SAME builder -> shared model nodes fuse
        terms = [
            gl.term(g, xin) * coeffs.get(key, 1.0)
            for key, gl in self.graph_losses.items()
        ]
        total = reduce(lambda a, b: a + b, terms)
        return g.build(total)(x, cache=cache)


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    torch.manual_seed(0)
    arch = GatedGraph(d_in=8, d_dict=16)
    tr = arch.make_trainable()
    x = torch.randn(32, 8)

    # the fused loss graph: print it and check the shared encoder runs once
    print("=== fused total-loss graph (gated: main + aux + 3 losses) ===")
    g = GraphDef()
    xin = g.input()
    terms = [
        gl.term(g, xin) * tr.coeffs.get(k, 1.0) for k, gl in tr.graph_losses.items()
    ]
    print(g.build(reduce(lambda a, b: a + b, terms)))

    CountLinear.total = 0
    total = tr.loss(x, cache=cl.Cache())
    ok("total loss is a scalar", total.dim() == 0)
    ok(
        "encoder ran ONCE across main + aux + all 3 losses (fused)",
        CountLinear.total == 1,
    )
    built = g.build(reduce(lambda a, b: a + b, terms))
    n_relu = sum(
        1
        for n in built._order
        if isinstance(n.producer, Op) and n.producer.fn is torch.relu
    )
    ok("enc_gate relu fuses across its 3 uses (only 2 relus: gate + mag)", n_relu == 2)

    # ...and it trains
    opt = torch.optim.Adam(tr.parameters(), lr=1e-2)
    history = []
    for _ in range(150):
        loss = tr.loss(x, cache=cl.Cache())
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    ok("loss decreased", history[-1] < history[0])
    print(f"  loss: {history[0]:.4f} -> {history[-1]:.4f}")
    print("\nall checks passed.")


if __name__ == "__main__":
    main()
