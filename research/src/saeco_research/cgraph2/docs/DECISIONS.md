# cgraph2 — design decisions log

Append-only. Each entry: the call, the options, the choice, the reasoning, and
the alternative we didn't take. Significant calls only (model/API/semantics).

---

## D-01 — GraphSpec is the runnable context (fold in ValueGraph)

**Call:** Should the runnable + registration layer be separate (`ValueGraph`) from
the authoring layer (`GraphSpec`)?

**Choice:** No — make a `GraphSpec` instance itself runnable. It registers its own
producers (so it's a real `cl.Module`: `.parameters()`, `.to()`, `state_dict`),
and exposes `.forward_values` / `.collect` / `forward`. `ValueGraph` is retired
into it.

**Why:** Glen's "a ref only means something in relation to its graph." A handle is
a node, but to *run* it you need its producers registered on a module, and that
context is the graph. Bundling handles + registration + run into one object makes
"access the handle through its graph" literal and kills the `ValueGraph(output,
(extras...))` hand-assembly smell. Matches "everything is a graph."

**Mechanism (as built):** `GraphSpec` stays a *plain object* (not an nn.Module).
It runs off the *live* module references its nodes already hold (no registration
needed to compute), so it gets `forward_values`/`collect`/`forward` directly. For
the nn.Module bookkeeping a trainable needs (`.parameters()`/`.to()`/state_dict),
`g.register_into(module)` walks the cone and registers the leaf producers onto the
owning module. `GraphSpec.of(**handles)` builds one dynamically for runtime-sized
aggregates (the trainable's whole computation). `ValueGraph` retired.

**Why not nn.Module:** the original plan (make `_GraphMeta` ABCMeta-based,
`GraphSpec(cl.Module)`) collides head-on with `g(x)` = *apply* (D-02 composition):
`nn.Module.__call__` is *run*. Keeping `g(x)` = apply (so `model(x).acts`
composes) and separating "run" (`g.forward_values`) from "register"
(`g.register_into`) is cleaner and avoids the conflict. The trainable registers
by calling `self.graph.register_into(self)`.

---

## D-02 — Keep nested-subgraph structure (don't flatten); dedup is the execution view

**Call:** When graph g1 composes into g2, is g1 flattened away, or kept?

**Choice:** Kept. g2 keeps its bound sub-graphs as navigable children (`g2.g1`,
`g2.g1.n`), preserving the boundary. The globally-interned node DAG is the
*execution* view (interning gives the dedup/fusion); the nested children are the
*structural* view. They coexist and are consistent (children's handles point into
the interned DAG).

**Why:** Glen: "keep the whole subgraph in the representation, really... not just
leave a trace." Inlining-vs-bubble-up was a false tension — keep structure for
navigation/bubble-up, derive the deduped DAG for running.

**Note on mechanism:** Glen described computing the deduped path at *compute* time
and caching it. We get the same deduped DAG eagerly via interning at construction
(equivalent result, already "cached" by construction). Recorded as a possible
future change if eager interning ever gets in the way; not needed now.

---

## D-03 — Bubble-up metrics (not model-exposed handles)

**Call:** Metrics (L0/L1) as (a) bubbled-up sites collected from the graph, or
(b) plain handles the model graph exposes by name.

**Choice:** (a), bubble-up — Glen: "I do not like B, I like bubble up metrics
being available." A site (e.g. `ActMetrics`) lives on `acts` wherever the model
puts it; the trainable `collect`s it from the graph cone and derives its handles
from the collected node's (already-rebound) input, so the metric is computed in
the full normalized context for free.

**Why:** information bubbles up from where it lives without the outside wiring it —
the saeco ActMetrics property, made graph-native. Reserves the machinery for the
stateful case too (freq tracker's deferred update).

**Alternative:** (b) was simpler but Glen explicitly wants bubble-up.


>> to give more context on this, if the only metrics under consideration were L0 and L1 then (b) would totally be superior. But I think it's important to be able to collect and log metrics from the inside that we can't plan for from the outside, so as I see it there needs to be some sort of contract that enables passing up what needs to be logged from the inside to the outside. 
---

## D-04 — Total loss is a fused graph handle (weights as scalar source-nodes)

> **SUPERSEDED (overnight self-review).** The weight *buffers* are unsafe across
> devices: `nn.Module._apply` (`.to(device)`) **reassigns** a buffer to a new
> tensor, so the `ref(w, ())` source-node (captured at init) and `self._weights[k]`
> orphan on CPU while the registered buffer moves to CUDA — verified on GPU
> (registered buffer on cuda, node's tensor on cpu, a dead duplicate in
> state_dict; it only "worked" via silent `cuda * cpu-0dim` promotion). Reverted to
> **device-safe Python weighting** (`total = sum(term * coeff)`): the loss terms
> still share one fused `forward_values` pass (the fusion that matters), and the
> dynamic coeff weight is applied in Python. A genuine `total_loss` *handle* needs
> weights as real graph **inputs** — awaits multi-input/`DictBatch`. Lesson: a
> graph node captures a *tensor*; only `nn.Parameter` survives `.to()` in place, so
> graph-referenced state must be a param or a module that reads its own live buffer,
> never a bare registered buffer.


**Call:** Is the weighted total loss a Python sum of saved terms, or a handle the
graph computes?

**Choice:** A handle. Each loss term is multiplied by a per-loss scalar weight
*source node* (`ref(weight_buffer, ())`), summed into `total_loss`. `loss()`
syncs the weight buffers from `coeffs` (so L0-targeting's dynamic sparsity weight
flows in), runs one `forward_values` pass, and `save(total_loss)` is the backprop
loss; the *unweighted* terms are also saved for logging (matching saeco's
`cache.<lossname>` = unweighted, `cache.loss` = weighted sum).

**Why:** Glen: "we have some input that's the weights on the losses, then we
multiply each by its corresponding loss and we sum" — the total loss *is* a graph.
Execution-equivalent to Python weighting (the terms share one fused pass either
way), but `total_loss` becomes a first-class handle, which is the graph-native form.

**Note:** weights are scalar *buffers* synced each step — a pragmatic stand-in for
true graph *inputs*. The cleaner form passes the weight vector as an input node,
which awaits multi-input/`DictBatch`. Recorded for later.

---

## D-05 — ActMetrics is a GraphSpec; a thin site marker makes it bubble-up-able

**Call:** "Make ActMetrics a GraphSpec" (Glen) vs. it must be `collect`-able as a
site. Tension: a `GraphSpec` *inlines* when composed (no marker node to collect),
and a `GraphSpec` instance can't be a passthrough producer (its `__call__` is
apply, not identity).

**Choice:** Split the two roles. `ActMetrics(GraphSpec)` is the reusable metric
*computation* (`acts -> L0/L1`, passthrough `output`). A thin `_MetricsSite`
producer (real passthrough) is what gets spliced onto `acts` (`metered(acts)`) and
`collect`ed; it carries the metrics graph class and, on collection, applies it to
the *rebound* acts (`site.at(node.inputs[0])`) — so the metric is computed in the
full normalized context (the site node sits on the recon path, so it rebinds with
the model). Metrics thus bubble up (D-03): the model buries a site on its acts, the
trainable `collect`s it, and nothing in between declares L0/L1.

**Why:** keeps the metric a clean composable graph *and* collectable *and*
full-context. The site marker is plumbing; `ActMetrics` is the graph.

**Alternative not taken:** make the GraphSpec instance itself the producer
(breaks: `__call__` = apply) or tag the passthrough with `self` in the class body
(breaks: no instance at class-def time).

---

## D-06 — Review-pass fixes (self-review + codex)

A focused codex review (of core.py/spec.py/sae.py/gated model.py) plus self-review
surfaced and fixed:

- **P1 arg-slot identity (spec.py).** `arg(default=Input)` returned the *global*
  `Input` as the slot's identity, so binding such a slot would `subs[Input]=...` and
  rewrite **every** `Input` use (and any other default-Input slot collides). Fixed:
  every `arg()` is a unique `_ArgSlot(default)` marker node; the unbound slot
  resolves to its default at construction. Regression test added.
- **P1 deferred ordering (core.py).** `_SaveSession.__exit__` ran each deferred
  callback immediately after computing its value, so an earlier side-effect could
  mutate state a later deferred target reads; and the contextvar wasn't reset on
  exception. Fixed: materialize all deferred values first, then run callbacks, only
  on clean exit, with `_ACTIVE_SESSION.reset()` in `finally`.
- **P1 weight buffers** — see D-04 (already reverted to Python weighting).
- **P2 sub_slot inheritance (spec.py).** A subclass overriding a `sub_slot` didn't
  re-point the base's `_SubGet`s. Fixed: re-point sub-slots in the inheritance
  merge, and `_SubGet.__graph_rebind__` now distinguishes a marker (re-point) from a
  bound bundle (splice).
- **P2 register_into bare tensors (core.py).** Now registers bare buffer producers
  via `register_buffer` (with the D-04 staleness caveat documented).
- **P2 empty-loss guard (sae.py).** `GraphTrainable` raises if `losses` is empty.

All cgraph2 demos + 9 pytest + ruff + GPU green after the fixes.

---

## D-07 — `AffineNormalizer` is the *static collapse* of `GNConfig`; dynamic aggs raise

**Call:** Glen's TODO — make the normalizer consume a `GNConfig` and match
`GeneralizedNormalizer`. How much of `GNConfig` does the clean graph-native
normalizer cover, and what happens for the configs it doesn't?

**Choice:** `AffineNormalizer(d_data, cfg=GNConfig(), eps=1e-7)` consumes a
`GNConfig` and covers exactly the **static** aggregations `{DONTUSE, PRIMED}` for
all four terms (`mu_e`/`mu_s`/`std_e`/`std_s`). It primes by replaying
`GeneralizedNormalizer`'s sequential estimation (`mu_e -> mu_s -> std_e -> std_s`,
each on the running-normalized data) and **folds the four factors into two primed
buffers** `mu = mu_e + mu_s`, `scale = std_e * std_s` — so `forward = (x-mu)/scale`
and `invert = x*scale + mu` are one affine map reading its own buffers (device-safe;
no `mu_e(x)` cache-routing methods, no aggregation dispatch at run time). The
dynamic aggregations (`RUNNING_AVG`, `BATCH_AVG`, `LEARNED`, `SAMPLE`) and
`sandwich` raise `NotImplementedError` at construction, pointing back to the class.

**Why:** the whole point of `StaticInvertibleGeneralizedNormalizer` is that, for
static configs, the normalizer *is* a fixed affine map — so represent it as one
(two buffers), not as a config-dispatching machine. That's the "small elegant
interface over intricate machinery" principle. Matching the stock normalizer to
<1e-5 (verified on the default cfg and `mu_e=PRIMED`) makes it a drop-in for the
configs people actually sweep with the static-invertible normalizer. The four
dynamic terms genuinely *aren't* a static affine map: BATCH_AVG/SAMPLE recompute
per batch (need cache routing / a per-batch handle), RUNNING_AVG is a stateful EMA
(the same bubble-up shape as the FreqTracker — a future graph component), LEARNED
is a trained parameter. Failing loud at the boundary (principle 4) is the right
call rather than silently approximating them with a frozen primed value.

**Alternative not taken:** port the full `mu_e(x)`/`std_e(x)` cache-routed
machinery into the graph normalizer. That re-imports the complexity the static
collapse exists to remove; the dynamic cases are better served as their own graph
components (deferred, per the bubble-up pattern) than as enum branches inside the
affine map.

All cgraph2 demos + 11 pytest + ruff + GPU (real Trainer, trains identically) green.

---

## D-08 — `field()`: the multi-input primitive (built); the migration is proposed

**Call:** The single global `Input` node means a graph reads only one runtime
value — blocking a true `total_loss` handle (weights as inputs) and transcoders
(`y_norm != x_norm`). How to add multi-input, and how much to do autonomously?

**Choice:** Build the *primitive* now, propose the *migration*. `field(name)` is a
stateless `_Field` producer node `ref(_Field(name), (Input,))`, interned per name,
that extracts a named field off a **structured input** seeded at `Input` (`dict` →
subscript, else `getattr`; fail-loud on a missing field). It's the Option-B
("one structured input / DictBatch") surface over the low-level named-input
mechanism. Purely additive: the bare-tensor `Input` path is untouched, so every
existing graph, the 5 prior demos, and the trainable are unchanged; a graph either
reads `Input` as the value or reads `field(...)` off a struct. Shipped with
`multi_input_demo.py` (a transcoder-shaped weighted loss reading `input` /
`target` / `weight` as three fused fields). Full design + the deferred migration
in `docs/DESIGN-multi-input.md`.

**Why:** the primitive is near-zero-risk and proves the foundation (fusion +
device-clean, since fields are plain extraction nodes, not the stale buffers that
killed D-04). The *migration* — `NormalizedModel` reading a `target` field, the
trainable packing `coeffs` into the batch to expose a `total_loss` handle — changes
working behavior and has real open questions (DictBatch type, per-loss vs vector
weights, missing-`target` policy), so per "let done mean done" it awaits Glen's
greenlight rather than a guessed autonomous migration.

**Alternative not taken:** named positional input nodes (`input_node("y")`) seeded
via extra `forward_values` args. That's the same mechanism with a positional-soup
surface; the struct/`field` surface matches saeco's already-structured
`SAETrainBatch` and adds fields without changing any graph that doesn't read them.

All cgraph2 demos (now 6) + 13 pytest + ruff green; existing GPU run unaffected
(the primitive isn't on the trainable's path yet).

---

## D-09 — Typed `DictBatch` contracts (`field(Batch.input)`, no string accesses)

**Call:** Glen wants the multi-input surface *typed* — `field(Batch.input)`, not
`field("input")` — with typed batch construction, an override escape hatch, and
converters. His worry: is the typed surface expressible to the type checker?

**Choice:** Yes, and built (`dictbatch.py`). A `DictBatch` subclass declares fields
as annotations; a `_BatchMeta` metaclass makes each *class-level* field a typed
`FieldRef` (pyright reports the annotation type — `Batch.input: Tensor`, verified by
probe), while a *constructed* `Batch(input=…)` exposes the values as instance
attributes. `dataclass_transform` gives typed construction (pyright checks field
types + missing fields). A declared-but-unprovided field fails loud on instance
access (a `__getattribute__` guard catches the class-`FieldRef` fallback that would
otherwise be a silent wrong value). `field()` duck-types `str | FieldRef`, so
`field("x")` and `field(Batch.x)` coincide. `dictbatch.py` + the typed demo are
pyright-clean on the user surface.

- **Override** (Glen's "not forced onto the dictbatch input"): no new mechanism —
  `x: Arg = arg(default=field(Batch.input))` makes the field an overridable slot;
  bind `x` at construction to bypass the batch. Just `field` + `arg` composed.
- **Converters** (Glen's "throw a converter in"): no new mechanism either — a
  converter is a typed function `Source -> InputContract` (the graph's input
  `DictBatch`). Verified: one graph runs under an autoencoder converter
  (`target=input`) and a transcoder converter (`target≠input`). This dissolves the
  "baked-in unpacking vs converter-on-front" fork — the graph names *its own input
  contract's* fields (its interface) and converters adapt external data to it.
- **Multi-output / logging** (Glen's dual + his D-03 note): proposed, not built —
  an output recipe `{field -> ValueRef}` producing a typed output `DictBatch`, with
  logging as a *dynamically* bubble-up-built recipe (`collect`), and
  `@DictBatch.auto_other_fields` as the typed-core + dynamic-tail contract for
  "metrics you can't plan from outside." Design in `docs/DESIGN-multi-input.md`.

**Why:** typed contracts are what make the batch a real interface (catch wrong
types / missing fields at author time), and the metaclass "lie" (class field typed
as its value type, runtime `FieldRef`) is the same pattern the codebase already uses
(`Swept[T]`, the GraphSpec slots). The dynamic logging tail is the one place static
typing is rightly surrendered — those metrics are by definition not known outside.

**Alternative not taken:** string-keyed `field("input")` only (untyped, rename-
invisible — against the repo's "don't restate a symbol as a string" convention);
abstract input slots + a separate converter-into-slots mechanism (more machinery
than function composition into a typed contract, and harder to type).

14 pytest + 6 demos + ruff green; `dictbatch.py` + demo pyright-clean on the user
surface. Trainable migration (D-08's open items) still pending greenlight.

---

## D-10 — Trainable migration: total_loss is a graph handle; transcoder target

**Call:** With multi-input primitives built (D-08/D-09), migrate the trainable off
Python loss-weighting + hardcoded `y_norm = x_norm`.

**Choice:** Done, on the existing `field`/`bind`/`forward_values` machinery — no new
mechanism (per refinement 2, this is all wiring):

- `NormalizedModel` reads `field("input")` and `field("target")`; `y_norm =
  normalize(field("target"))` (was `= x_norm`). Transcoders (`y != x`) fall out;
  an autoencoder just seeds `target == input`.
- `GraphTrainable` builds `total_loss = Σ field("w_<k>") * term_k` as a **first-class
  graph handle**. `loss()` seeds a boundary batch `{input, target, w_<k>}` (weights
  from `coeffs`), runs one `forward_values`, and `save(total_loss)` is the backprop
  loss. The terms are still saved unweighted for logging.
- Inference paths (`forward`/`get_acts`) seed `{"input": x}` (fail-loud, not a
  silent bare-tensor fall-through).

**Why this is the clean D-04 resurrection:** the loss weights are now graph
**inputs** (seeded fresh from `coeffs` every step), not captured buffers — so there
is nothing for `.to(device)` to orphan. The dynamic L0-targeting sparsity weight
flows in as the `w_sparsity_loss` field. Validated: GPU through the real Trainer
trains identically (L2 ~0.02, cosim ~0.99, L0-targeting bends L0), 16 pytest + 6
demos + ruff green, plus a focused test asserting `total == Σ coeff·term` and that
a `target != input` changes the recon loss.

**Still deferred:** the bubble-up *logging* recipe (the output-side dual) — the
cache-write is still hand-listed in `loss()`. That's the next slice when we want
inside-metrics to appear without the outside naming them (D-03).

---

## D-11 — `sub_slots()` + deferred `.reduce()`: dynamic folds in a class block

**Call:** Glen wants the trainable's outer graph (incl. the `total_loss` reduce over
a *dynamic* set of losses) expressed as a real `GraphSpec` **class block**, not the
imperative `GraphSpec.of(...)`. The blocker: a class body runs at class-def, but the
loss collection is bound at construction — you can't iterate a slot that isn't
bound yet (`for k, loss in losses.items()` can't work in the body).

**Choice:** A variadic sub-model slot `sub_slots()` (bound to a `dict` of
sub-graphs) plus a **deferred fold**: `losses.reduce(per_item, combine)` returns a
`_MapReduceOverSlots` node whose `__graph_rebind__` — at construction, when the
slot is bound — runs `per_item(key, subgraph)` for each bound sub-graph and folds
with `combine`. So the class body is static and declarative:

    class TrainGraph(GraphSpec):
        losses: dict[str, GraphLoss] = sub_slots()
        main: NormalizedModel = sub_slot()
        total_loss = losses.reduce(lambda k, l: field(f"w_{k}") * l.out, reduce_sum)
        output = main(Input).out

The iteration is *deferred* to construction and re-interned there, so the result is
one flat N-ary node (via `reduce_sum`), as fused/deduped as any static graph. The
bound dict is a navigable child (`g.losses["L2_loss"]`). The trainable now builds
its graph as `TrainGraph(losses=loss_graphs, main=main_nm)`.

**Why:** this is the declarative class-block form Glen asked for, and it's the
minimal mechanism for it — `.reduce(per_item, combine)` is a deferred map-reduce
HOF (the per-item lambda is the template, run per bound item at construction),
reusing the existing `__graph_rebind__` splice path rather than a new template
engine. It generalizes beyond total_loss (any fold over a dynamic sub-graph set).

**Alternative not taken:** keep building the reduce imperatively in the trainable
(`weighted_sum(self._loss_terms)` via `GraphSpec.of`) — works, but buries the
structure in trainable code instead of a legible class block; and a full
`branch_over` template engine — more machinery than the deferred-reduce HOF needs.

Validated: 17 pytest (+ a focused `sub_slots` test) + 6 demos + ruff + GPU through
the real Trainer trains identically. `weighted_sum` stays as an exported component
for the imperative case; `reduce_sum` is the fold combiner.

---

## D-12 — the normalizer embeds its own inverse (partial sub-graph), so the model never names scale/shift

**Call:** How should `NormalizedModel` denormalize the model's output? The inverse
of an affine normalizer isn't a plain `ValueRef` you can write down up front: you
can't invert until the forward has produced `scale`/`shift`. The old wiring leaked
this — `NormalizedModel` took the normalizer's `denorm` as a second `op_slot` and
applied it (`out = denorm(y_pred_norm)`), so the model had to know the normalizer
exposes an inverse op.

**Choice:** "B-full" — embeddable, partially-applied sub-graphs. A `Denormalizer`
GraphSpec is the inverse map `value * scale + shift`; the normalizer embeds it
*partially applied* over its own primed factors, leaving `value` open:

    class GraphNormalizer(GraphSpec):
        affine: Op = op_slot(); scale: Arg = arg(); shift: Arg = arg()
        normalized  = affine(Input)
        denormalize = Denormalizer.partial(scale=scale, shift=shift)   # value open
        output      = normalized

    class NormalizedModel[T: SAEModel](GraphSpec):
        model: T = sub_slot(); normalizer: GraphNormalizer = sub_slot()
        nx      = normalizer(x_in)
        applied = model(nx.normalized)
        out     = nx.denormalize(applied.recon).output     # complete the partial

So `NormalizedModel` plugs the normalizer into a `sub_slot`, and completes its
embedded partial with the model's recon — the normalizer **owns its own inverse**
and the model never references scale/shift. `nx.denormalize(recon)` resolves at
construction (once `normalizer` is bound) to the embedded partial rebound to the
applied input, completed on `recon`, exposing `.output`. The decode path uses the
standalone form `normalizer.denormalize_value(acts→Input)`. `scale`/`shift` are
0-ary ops reading the `AffineNormalizer`'s live buffers, stably consed by
`id(normalizer)` so forward + inverse fuse. `GraphNormalizer.of(affine)` wires it;
`primed`/`prime_normalizer` stay passthroughs to the wrapped `AffineNormalizer`
(the trainer primes through them).

**Why:** the inverse genuinely depends on the forward's factors, so the principled
shape is a sub-graph closed over those factors with one open input — exactly what
partial application + embedding gives. It removes the cross-component leak (the
model no longer needs to be handed an inverse op) and the mechanism (Steps 1/2a:
`GraphSpec.partial` → `_PartialGraph`; `nx.attr(value).output` via
`_SubGet.__partial_call__`/`_AppliedPartial`/`_PartialComplete`) is general — any
GraphSpec can embed a partially-applied sub-graph and complete it through a
`sub_slot` application, not just the normalizer.

**Alternative not taken:** keep the two-`op_slot` `normalize`/`denorm` wiring — it
works but leaks the inverse across the component boundary and can't express "invert
through the factors this forward produced" for a non-static normalizer; and a
bespoke "invertible module" protocol — more special-casing than the general embed
mechanism, which we wanted anyway (Glen: "being able to embed subgraphs like that
could be very useful in other situations… the principled solution").

Validated: 21 pytest (incl. `test_graph_normalizer_embeds_its_inverse` — round-trip
+ `AffineNormalizer.invert` identity + consing/distinctness — and the gated-graph
trainable tests) + 6 demos + ruff clean + GPU through the real Trainer trains
identically (L2 ~0.02, cosim ~0.99, L0 ~3300 then falling).

---

## D-13 — deferred side-effects are training callbacks: `@deferred_execution` + a session `training` gate

**Call:** A stateful site (the freq tracker) writes its EMA back *after* the pass —
so the handle it reads is the pre-update state — by deferring onto the
`forward_values` session. The first cut put that logic *inside* `FreqTracker.update`:
it reached into `_ACTIVE_SESSION`, and `update` dispatched on its own argument type
(a `ValueRef` meant "schedule me", a tensor meant "run me"), re-entering itself at
block exit. Two problems: (1) the self-dispatch-on-arg-type is a confusing overload;
(2) `Trainer.eval_step` calls `loss` under `.eval()`, and the loop advanced the EMA
unconditionally, contaminating the dead-feature freqs that resampling / the `below_*`
metrics read.

**Choice:** make "deferred, training-only" a first-class session concept.
- `_SaveSession` carries a `training: bool` (passed via `forward_values(...,
  training=...)`; the trainable passes `self.training`). On clean exit it runs each
  deferral, **skipping training-only ones when `not training`** — still two-phase
  (materialize every deferred value in the clean memoized pass, then run the side
  effects), preserving the earlier ordering fix.
- `@deferred_execution` (a decorator) marks a method as a training callback: inside
  a session it's collected and run at block exit with any `ValueRef` args
  materialized; outside a session it runs immediately. `FreqTracker.update` becomes a
  plain `self.freqs.copy_(new_freqs)` under the decorator — no `_ACTIVE_SESSION`, no
  arg-type dispatch.
- `save.after(handle, callback)` stays the low-level, *always-run* primitive (an
  arbitrary one-off callback); `@deferred_execution` is the training-only,
  method-level surface.

**Why:** Glen asked for "a way to collect *training* callbacks that get executed
inside the `with forward_values` block", and to either let the context manager manage
the deferral or at least have an explicit `deferred_execution` decorator. This does
both: the *session* owns the deferral + the training gate (the context manager
managing it), and the *decorator* is the explicit, declarative registration surface —
so the side-effecting method reads like a normal method and the deferral / eval-gating
is out of its body. It fixes the eval-contamination bug as a property of the
mechanism, not a one-off `if self.training`.

**Alternative not taken:** a bare `if self.training:` guard around the freq loop in
`loss` — fixes the symptom but leaves the `_ACTIVE_SESSION` self-dispatch in `update`
and doesn't generalize; and fully auto-collecting every site's update at block exit
(no explicit call in `loss`) — cleaner still, but that belongs to the freq-tracker
migration (it needs the site/handle wiring settled first), so the decorator is the
right increment now. `@deferred_execution` is training-only with no opt-out for
simplicity (YAGNI); add a flag if a non-training deferred method ever appears.

Validated: 58 pytest (incl. `test_deferred_execution_training_gate`,
`test_freq_tracker_eval_pass_does_not_advance_ema`,
`test_batch_honors_transcoder_target`) + 6 demos + ruff clean + GPU trains
identically.
