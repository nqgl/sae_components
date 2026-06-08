"""The gated SAE architecture on cgraph v2 (relational / interned).

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph2.gated

Same structure as the v1 port (`cgraph/gated_graph.py`), but on the relational
core: there is no builder and no `apply(g, x)` -- the model is just *interned
nodes* that read the global `Input`. `enc_gate` is one node by construction, so
referencing it from the main model, the aux model, and the sparsity loss shares
it automatically. The total loss is built once; the encoder runs once.
"""

from __future__ import annotations

from functools import reduce

import torch
import torch.nn as nn
from attrs import define

import saeco.core as cl
from saeco.trainer.trainable import Trainable
from saeco_research.cgraph2 import Input, ValueRef, op, ref
from saeco_research.cgraph2.core import Op  # the runtime Op class (for isinstance)


class CountLinear(cl.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=bias)
        self.calls = 0

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        self.calls += 1
        return self.lin(x)


class Linear(cl.Module):
    def __init__(self, d_in, d_out, bias=True):
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


@define
class SAEH:  # a model's handles (recon + the penalized acts)
    recon: ValueRef
    acts: ValueRef


@define
class GraphLoss:  # ~ Loss(model): wraps a model's handles, emits a loss-term node
    model: SAEH

    def term(self) -> ValueRef:
        return self.compute(self.model)

    def compute(self, m: SAEH) -> ValueRef:
        raise NotImplementedError


class GraphL2Loss(GraphLoss):
    def compute(self, m):
        return op(_mse, label="L2")(Input, m.recon)


class GraphSparsityLoss(GraphLoss):
    def compute(self, m):
        return op(_l1, label="L1")(m.acts)


class GatedV2(nn.Module):  # ~ the Gated arch; a container of parts
    def __init__(self, d_in: int, d_dict: int):
        super().__init__()
        self.encoder = CountLinear(d_in, d_dict)  # shared by enc_mag + enc_gate
        self.dec_main = Linear(d_dict, d_in)
        self.dec_aux = Linear(d_dict, d_in)
        self.gate_bias = nn.Parameter(torch.zeros(d_dict))
        self.mag_bias = nn.Parameter(torch.zeros(d_dict))
        self.log_r = nn.Parameter(torch.zeros(d_dict))

    def _h(self) -> ValueRef:
        return op(self.encoder)(Input)  # interned -> the SAME node every call

    def enc_gate(self) -> ValueRef:
        return op(torch.relu)(self._h() + ref(self.gate_bias, ()))

    def enc_mag(self) -> ValueRef:
        exp_r = op(torch.exp)(ref(self.log_r, ()))
        return op(torch.relu)(self._h() * exp_r + ref(self.mag_bias, ()))

    def gated_model(self) -> SAEH:  # ~ @model_prop
        gate = op(_thresh, label="Thresh")(self.enc_gate())
        acts = gate * self.enc_mag()
        return SAEH(recon=op(self.dec_main)(acts), acts=acts)

    def model_aux(self) -> SAEH:  # ~ @aux_model_prop
        gate_acts = self.enc_gate()  # SAME node as the main model's gate
        return SAEH(recon=op(self.dec_aux)(gate_acts.detach()), acts=gate_acts)

    def losses(self) -> dict[str, GraphLoss]:  # ~ the @loss_prop set
        return {
            "L2_loss": GraphL2Loss(self.gated_model()),
            "L2_aux_loss": GraphL2Loss(self.model_aux()),
            "sparsity_loss": GraphSparsityLoss(self.model_aux()),
        }

    def make_trainable(self) -> GraphTrainable:
        return GraphTrainable(
            self, self.losses(), coeffs={"L2_aux_loss": 1.0, "sparsity_loss": 3e-4}
        )


class GraphTrainable(Trainable):  # ~ Trainable, fused-graph loss
    def __init__(self, arch: GatedV2, losses: dict, coeffs: dict):
        cl.Module.__init__(self)
        self.arch = arch
        self.graph_losses = losses
        self.coeffs = coeffs

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return self.arch.gated_model().recon.build()(x, cache=cache)

    def loss(self, x, *, cache: cl.Cache, y=None, coeffs=None):
        coeffs = {**self.coeffs, **(coeffs or {})}
        terms = [
            gl.term() * coeffs.get(key, 1.0) for key, gl in self.graph_losses.items()
        ]
        return reduce(lambda a, b: a + b, terms).build()(x, cache=cache)


def ok(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    torch.manual_seed(0)
    arch = GatedV2(d_in=8, d_dict=16)
    tr = arch.make_trainable()
    x = torch.randn(32, 8)

    print("=== fused total-loss graph (gated: main + aux + 3 losses) ===")
    terms = [
        gl.term() * tr.coeffs.get(k, 1.0) for k, gl in tr.graph_losses.items()
    ]
    built = reduce(lambda a, b: a + b, terms).build()
    print(built)

    arch.encoder.calls = 0
    total = tr.loss(x, cache=cl.Cache())
    ok("total loss is a scalar", total.dim() == 0)
    ok("encoder ran ONCE across main + aux + all 3 losses", arch.encoder.calls == 1)
    n_relu = sum(
        1
        for n in built._order
        if isinstance(n.producer, Op) and n.producer.fn is torch.relu
    )
    ok("enc_gate relu fuses across its uses (only 2 relus: gate + mag)", n_relu == 2)

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
