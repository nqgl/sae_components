"""SKETCH (not runnable) — field-binding composition & the multi-output recipe.

A guided tour to sync our mental models. The thesis: **converters, total_loss, and
bubble-up logging are all the same operation — composing graphs by binding
field-reads to producer handles.** Inputs and outputs are duals.

Read top to bottom. PART 0 is the *typing picture* (stubs, bodies elided). PARTS
1–6 are worked examples; each says what it shows and why it matters. `BUILT` marks
what already works today; `PROPOSED` marks the new pieces we're designing.

The one idea to take away: a graph is a typed arrow `InContract -> OutContract`,
and composition is typed arrow composition. The DictBatch contracts are the types;
the field *names* are the structural glue that interns/fuses the wiring.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

# ============================================================================
# PART 0 — the typing picture (stubs)
# ============================================================================


class ValueRef: ...  # an interned graph node = a handle (BUILT)


class Cache: ...


def op(fn: Any) -> Any: ...  # wrap a callable/module as an applyable producer (BUILT)


def field(spec: Any) -> ValueRef:  # BUILT: field(Batch.x) or field("x")
    """Read a named field off the structured input. `spec` is a typed `Batch.x`
    (reads as the field's value type) or a string. Returns a handle."""
    ...


# class DictBatch:  # BUILT (dataclass_transform'd; see dictbatch.py)
#     """A typed batch contract. `class B(DictBatch): x: Tensor` makes `B.x` a typed
#     FieldRef at the class (reads as `Tensor`) and the value at the instance.
#     `B(x=...)` is typed construction."""

#     def __init__(self, **fields: Any) -> None: ...

#     @classmethod
#     def auto_other_fields(cls, c: type) -> type:  # PROPOSED
#         """Decorator: the batch carries its declared (typed) fields PLUS arbitrary
#         extra ones — the typed-core + dynamic-tail contract the logging case needs
#         (the bubbled-up metrics aren't known at author time)."""
#         ...

from saeco.data.dict_batch import DictBatch


# The keystone: a graph is a typed arrow InContract -> OutContract.
# Today's GraphSpec is ~ GraphSpec[Tensor, Tensor]; the multi-in/out work makes
# both ends typed DictBatch contracts. (PEP 695 generics; we're on 3.13.)
class GraphSpec[InT, OutT]:
    out: ValueRef  # BUILT: the default single output handle
    produces: type[OutT]  # PROPOSED: the output contract (a DictBatch type)

    def forward(self, batch: InT, *, cache: Cache) -> Tensor:  # BUILT: runs `out`
        ...

    # PROPOSED — the multi-output recipe: run once, return the whole typed output
    # batch (every handle named like an OutT field populates it, one fused pass).
    def produce(self, batch: InT, *, cache: Cache) -> OutT: ...

    # PROPOSED — field-binding composition: stick `converter` on the front by
    # binding each `field(InT.f)` read in this graph to the converter's `f` handle
    # (by name). Typed arrow composition: (Root->InT) then (InT->OutT) = Root->OutT.
    def with_inputs[RootT](
        self, converter: GraphSpec[RootT, InT]
    ) -> GraphSpec[RootT, OutT]: ...


# ============================================================================
# PART 1 — typed input contract + field   (BUILT — this is D-09, runs today)
# ============================================================================


class Inputs(DictBatch):  # the graph's INPUT CONTRACT
    x: Tensor
    target: Tensor  # distinct from x -> transcoders


class Model(GraphSpec[Inputs, "Encoded"]):  # reads Inputs, will produce Encoded
    x = field(Inputs.x)  # typed: Inputs.x reads as Tensor
    target = field(Inputs.target)
    acts = op(...)(x)  # pretend: encoder
    recon = op(...)(acts)  # pretend: decoder
    loss = op(...)(recon, target)  # pretend: mse
    output = loss  # the default single `out`


# usage today:  Model().forward(Inputs(x=acts, target=acts), cache=Cache())
# `Inputs(...)` is typed construction; `Model.x`/`Model.target` read the fields.


# ============================================================================
# PART 2 — the multi-output recipe   (PROPOSED)
#   a graph doesn't just have one `output`; it can PRODUCE a typed DictBatch.
# ============================================================================


class Encoded(DictBatch):  # the OUTPUT CONTRACT
    acts: Tensor
    recon: Tensor
    loss: Tensor


class GraphSpec2[InT, OutT]:
    out: ValueRef  # BUILT: the default single output handle
    produces: OutT  # PROPOSED: the output contract (a DictBatch type)

    def forward(self, batch: InT, *, cache: Cache) -> Tensor:  # BUILT: runs `out`
        ...

    # PROPOSED — the multi-output recipe: run once, return the whole typed output
    # batch (every handle named like an OutT field populates it, one fused pass).
    def produce(self, batch: InT, *, cache: Cache) -> OutT: ...

    # PROPOSED — field-binding composition: stick `converter` on the front by
    # binding each `field(InT.f)` read in this graph to the converter's `f` handle
    # (by name). Typed arrow composition: (Root->InT) then (InT->OutT) = Root->OutT.
    def with_inputs[RootT](
        self, converter: GraphSpec[RootT, InT]
    ) -> GraphSpec[RootT, OutT]: ...


@DictBatch.auto_other_fields
class Encoded2(DictBatch):  # the OUTPUT CONTRACT
    acts: Tensor
    recon: Tensor
    loss: Tensor


class Model2(
    GraphSpec2[Inputs, Encoded2]
):  # problem that  I don't know if we can link Encoded2 to here since it's in the body
    x = field(Inputs.x)  # typed: Inputs.x reads as Tensor
    target = field(Inputs.target)
    acts = op(...)(x)  # pretend: encoder
    recon = op(...)(acts)  # pretend: decoder
    loss = op(...)(recon, target)  # pretend: mse
    output = loss  # the default single `out`

    @classmethod
    def select(cls) -> Encoded2:
        return cls(acts=Model2.acts, recon=Model2.recon, loss=Model2.loss)


class Ref: ...


from returns.primitives.container import BaseContainer
from returns.primitives.hkt import SupportsKind1
from typing import dataclass_transform
from returns.interfaces import bindable


@dataclass_transform()
def fake_dct(): ...


@fake_dct
class ContractBatch[T](
    BaseContainer, SupportsKind1["ContractBatch", T], bindable.Bindable1[T]
):
    acts: T
    recon: T
    loss: T


from returns.primitives.hkt import kinded, Kind1


@kinded
def contract_to_impl(
    inp: Kind1[ContractBatch, Ref],
) -> Kind1[ContractBatch, torch.Tensor]: ...


class Model3(GraphSpec2[Inputs, ContractBatch]):
    x = field(Inputs.x)  # typed: Inputs.x reads as Tensor
    target = field(Inputs.target)
    acts = op(...)(x)  # pretend: encoder
    recon = op(...)(acts)  # pretend: decoder
    loss = op(...)(recon, target)  # pretend: mse
    output = loss  # the default single `out`
    contract = ContractBatch[Ref](acts=acts, recon=recon, loss=loss)


r = contract_to_impl(Model3.contract)

# Model (above) declares `GraphSpec[Inputs, Encoded]` and has handles named
# `acts`, `recon`, `loss` — which match Encoded's fields. That mapping
# {Encoded.acts: Model.acts, Encoded.recon: Model.recon, ...} IS the output recipe.

out: Encoded = Model().produce(Inputs(x=..., target=...), cache=Cache())  # one pass
out.acts  # typed Tensor; out.recon; out.loss   <- the whole output, typed
# Why it matters: `produce` runs all named handles in ONE fused forward_values pass
# and hands back a typed batch. `forward` (single `out`) is just `produce(...).<the
# output field>`. This is `GraphSpec.of(**handles)` generalized to a *typed* result.


# ============================================================================
# PART 3 — field-binding composition: a converter is a GRAPH   (PROPOSED)
#   this is the crux of your dedup point.
# ============================================================================


class RawActs(DictBatch):  # the ROOT batch (what actually arrives at runtime)
    activations: Tensor


# A converter-as-graph: GraphSpec[RawActs, Inputs] — produces Inputs' fields.
class Autoencode(GraphSpec[RawActs, Inputs]):
    produces = Inputs
    a = field(RawActs.activations)
    x = a  # handle 'x'      -> Inputs.x
    target = a  # handle 'target' -> Inputs.target   == x  (the SAME node!)
    output = a


# compose: bind Model's field(Inputs.x) -> Autoencode.x, field(Inputs.target) ->
# Autoencode.target, BY NAME. Result is a typed arrow RawActs -> Encoded.
runnable: GraphSpec[RawActs, Encoded] = Model().with_inputs(Autoencode())
result: Encoded = runnable.produce(RawActs(activations=...), cache=Cache())

# WHY THIS MATTERS (your dedup point): because Autoencode makes `x` and `target`
# the SAME node, after binding Model's recon path and the loss's target are reading
# one node — so any shared computation FUSES (interning). The graph keeps its
# "one fused pass" guarantee through the converter.
#
# Contrast a converter expressed as a plain FUNCTION:


def autoencode_fn(raw: RawActs) -> Inputs:  # function-converter (BUILT-style)
    return Inputs(x=raw.activations, target=raw.activations)


Model().forward(autoencode_fn(RawActs(activations=...)), cache=Cache())
# This RUNS and is correct, but x and target are *distinct* field-read nodes in the
# graph (the function is opaque to it), so identical work on x and target is
# computed TWICE. Fine when there's no sharing to exploit; lossy when there is.


# ============================================================================
# PART 4 — "Inputs becomes a typed interface, not a runtime object"   (the subtle bit)
# ============================================================================

# In PART 3's `with_inputs` (graph-converter) path: there is NO `Inputs` instance at
# runtime. The `field(Inputs.x)` reads got REPLACED (bound) by Autoencode's nodes;
# the only runtime batch is the ROOT `RawActs`. `Inputs` survives purely as the
# *typed contract* that lets Model and Autoencode agree on field names + types
# (Model is GraphSpec[Inputs, ...], Autoencode is GraphSpec[..., Inputs]).
#
# In PART 1/PART 3's function path: an `Inputs` instance IS built at runtime (the
# function returns one) and seeded. Same contract, two modes:
#   - graph-converter  -> Inputs is compile-time-only; fusion preserved.
#   - function-converter -> Inputs is a real runtime object; simple, no fusion.


# ============================================================================
# PART 5 — the unification: converter / total_loss / logging are ONE mechanism
# ============================================================================

# 5a. CONVERTER = input-side field-binding (PART 3).  Model.with_inputs(converter).

# 5b. TOTAL_LOSS = a handle in a composed graph; the loss WEIGHTS are fields.


class LossWeights(DictBatch):  # weights arrive as a (typed) input field group
    l2: float
    sparsity: float


class TrainStep(GraphSpec["TrainInputs", "TrainLog"]):
    # the model + each loss compose in (sub-graphs); weights are read as fields, so
    # the weighted total is just a HANDLE -- device-clean (fields are extraction
    # nodes, not the stale buffers that killed D-04).
    l2 = op(...)(...)  # an L2 loss sub-graph's output handle
    sparsity = op(...)(...)  # a sparsity loss sub-graph's output handle
    w_l2 = field(LossWeights.l2)
    w_sp = field(LossWeights.sparsity)
    total_loss = w_l2 * l2 + w_sp * sparsity  # <- total_loss is a first-class handle
    output = total_loss


# loss(batch, coeffs) packs coeffs into the input batch as LossWeights fields;
# `save(total_loss)` is the backprop loss. No weight buffers, so no .to() staleness.

# 5c. LOGGING = output-side field-binding, built dynamically by BUBBLE-UP.


@DictBatch.auto_other_fields  # typed core (loss, L0) + dynamic tail (bubbled sites)
class TrainLog(DictBatch):
    loss: Tensor
    L0: Tensor
    # ...and whatever metric sites bubble up (below_3e-5, aux L0, ...) that we
    # CAN'T name from outside -- your D-03 point. They land in the 'other fields'.


class TrainInputs(DictBatch):
    x: Tensor
    target: Tensor
    weights: LossWeights


# The trainable's whole step is GraphSpec[TrainInputs, TrainLog]. Its OUTPUT recipe
# is built by bubble-up: `collect(...)` walks the graph, each site contributes a
# (name, handle); that set IS the output recipe; `produce(...)` executes it in one
# forward_values pass and returns the TrainLog. New inside-metrics appear in the log
# WITHOUT the outside naming them -- which is the whole point of bubble-up (D-03).
#
#   log: TrainLog = train_step.produce(train_inputs, cache=cache)
#   log.loss, log.L0, log.below_3e_5   # 'below_3e_5' is an auto-other-field

# So: input recipe (converter) and output recipe (logging) are the SAME field-
# binding machinery pointed in opposite directions; total_loss is just a handle
# that falls out once weights are fields. One mechanism, three uses.


# ============================================================================
# PART 6 — function-converters still belong AT THE EDGE
# ============================================================================

# At the true data boundary there's no internal sharing to fuse -- you're just
# adapting an external struct to a contract. A plain typed function is the right
# tool there (no graph needed):


def from_sae_batch(b: Any) -> TrainInputs:  # SAETrainBatch -> cgraph contract
    return TrainInputs(
        x=b.input, target=b.target, weights=LossWeights(l2=1.0, sparsity=3e-5)
    )


# Rule of thumb:
#   - graph-converter  when the conversion introduces *sharing the graph should fuse*
#     (autoencoder x==target; anything where two contract fields share a source).
#   - function-converter at the outermost edge (external batch -> cgraph contract),
#     where it's pure adaptation and there's nothing to fuse.


# ============================================================================
# PART 7 — RESOLUTION of the explicit-wiring / Contract[T] / HKT question
#   (all pyright-confirmed in scratch; this is the recommended shape)
# ============================================================================
#
# Glen's wishes: (a) define the output contract near the graph, (b) wire output
# fields EXPLICITLY (not "same-name lifting"), (c) ideally a typed C[Ref] ->
# C[Tensor] realize. The HKT worry was about (c) being generic over C.
#
# FINDING: we get (a), (b), and a typed realize WITHOUT higher-kinded types and
# WITHOUT `returns`. The trick: don't ask for ONE statically-typed realize generic
# over all C -- name the realized type AT THE GRAPH (OutT = Encoded[Tensor]) and let
# the runtime realize iterate fields generically. The only thing HKT would buy is
# the one generic arrow, which we don't need.
#
# The shape (every reveal_type below verified with pyright):
from typing import dataclass_transform as _dt


@_dt()
class Contract[T]:  # output recipes are generic over their value type
    def __init__(self, **fields: T) -> None: ...


class Encoded3[T](Contract[T]):  # define the fields ONCE; T is Ref or Tensor
    acts: T
    recon: T
    loss: T


class Model4(GraphSpec[Inputs, "Encoded3[Tensor]"]):  # OutT = the *realized* contract
    x = field(Inputs.x)
    target = field(Inputs.target)
    acts = op(...)(x)
    recon = op(...)(acts)
    loss = op(...)(recon, target)
    # (b) EXPLICIT wiring, in Model4's scope (acts/recon/loss visible here) -- this
    # is your `produce = Encoded(acts=acts, ...)` idea, and it FIXES the scoping
    # problem you hit (`acts: Tensor = acts` inside a nested class body can't see
    # Model's acts; building the recipe in Model's body can). Typed: fields are Ref.
    recipe = Encoded3[Ref](acts=acts, recon=recon, loss=loss)  # : Encoded3[Ref]


# realize is runtime-generic (iterate the recipe's fields, save each handle) and
# typed at the call site via OutT:
#   out = Model4().produce(Inputs(x=.., target=..), cache=..)   # : Encoded3[Tensor]
#   out.acts   # : Tensor   <- realized + typed, no HKT
#
# CAVEATS (small):
#   - Contract[T] is HOMOGENEOUS (all fields share T). That's exactly right for an
#     output recipe (every field is a handle that realizes to a Tensor: Ref->Tensor).
#     A heterogeneous batch (loss: Tensor, step: int) stays a plain DictBatch (no
#     Ref/Tensor duality needed -- you don't build a recipe from it).
#   - The link "recipe's C == OutT's C" (Encoded3 in both) is by convention -- you
#     write Encoded3[Ref] for the recipe and Encoded3[Tensor] as OutT. Enforcing it
#     statically is the part that WOULD need HKT; not worth it.
#
# WHY NOT `returns`: (1) not a dependency here, and pulling a monadic-HKT library in
# for one feature is heavy + unidiomatic for this codebase; (2) its Kind1/Bindable
# is single-inner-value *container* semantics -- a multi-field record would need a
# custom map-over-all-fields, fighting the library; (3) even done right it only buys
# the generic arrow we just showed we don't need. So: skip it.
