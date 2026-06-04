# Design proposal — multi-input graphs (DictBatch) & the `total_loss` handle

> **Read the bottom section first** ("Update after Glen's review") — it supersedes
> the original proposal below with what's now BUILT (typed `DictBatch` contracts,
> D-09), the resolved converter fork, and the multi-output/logging design. The
> sections below are the original overnight proposal, kept for history.

**Status:** the **`field()` primitive + typed `DictBatch` contracts are built,
tested, demoed** (D-08, D-09); the **trainable migration** (`total_loss` handle,
transcoder `target`, bubble-up logging) is the proposal awaiting greenlight.
Written + the primitive landed during the overnight run because this is the biggest
repeatedly-deferred item and it gates two things we keep wanting (a true
`total_loss` handle, transcoders).

What I built autonomously (safe, purely additive — zero existing behavior changed):
`field(name)` in `core.py` (a stateless `_Field` producer that reads a named field
off a structured input seeded at `Input`), exported, rendered (`field[name]`), with
`cgraph2/multi_input_demo.py` (a transcoder-shaped weighted loss reading three
distinct fields — `input` / `target` / `weight` — that all fuse into one pass) in
the CI suite. **What I deliberately did *not* do:** migrate `NormalizedModel` /
the trainable onto fields, or add the `total_loss` handle — that's the part with
the open questions below, and it changes working behavior, so it's yours to
greenlight. Below is the recommendation + mechanism + migration.

## Why

Two limitations both trace to **one global `Input` node** (`core.py:159`):

1. **A graph reads only one runtime input.** For a transcoder the autoencoding
   *target* `y` differs from the *input* `x` (`y_norm != x_norm`). Today
   `NormalizedModel` hardcodes `y_norm = x_norm` (`sae.py`), so transcoders can't
   be expressed.
2. **The weighted total loss can't be a graph handle.** D-04 tried weights as
   scalar source *buffers* and they went stale across `.to(device)` (a node
   captures a tensor; `.to()` reassigns buffers). We reverted to Python weighting.
   The clean fix is weights as real graph **inputs** — but there's only `Input`.

Both dissolve if a graph can read **several named runtime values**.

## The fork

- **Option A — named input nodes.** Generalize `Input` to a small interned
  registry: `input_node("x")`, `input_node("y")`, `input_node("w_sparsity")`,
  each a distinct producer. `forward_values(**inputs)` seeds them.
- **Option B — one structured input (DictBatch).** `Input` carries a struct; a
  graph reads fields via `field("x")`, `field("target")`, `field("weights")`,
  each interning a `ref(_Field(name), (Input,))` extraction node.

### Recommendation: **B (DictBatch), built on A's mechanism.**

Reasoning, in the house style:

- The training batch is **already a struct** — saeco's `SAETrainBatch` bundles
  `input` / `target` / … . "Everything is a graph" wants the graph's input to be
  that same struct, not a positional soup. `field("target")` reads what's
  semantically there; a graph that only needs `input` only references that field
  (and only that field is realized — fusion/dedup unchanged).
- It composes: x, y, masks, per-loss weights are all just fields. Adding a field
  never changes a graph that doesn't read it.
- A's named nodes are the right **low-level** primitive; `field(name)` is the
  ergonomic surface over them. So build A, expose B.

`bind`/`at` already substitute *any* node, so they generalize for free:
`bind(target, {field("x"): xn, field("target"): yn})` works the moment the field
nodes exist. `at(handle, x)` stays the single-input sugar (`{Input: x}`).

## Mechanism (sketch)

- `core.py` **(BUILT)**: `_Field(name)` producer + `field(name) -> ref(_Field(name),
  (Input,))`, interned per name (module-level `_FIELDS`). A `_Field` reads
  `batch[name]` for a dict / `getattr(batch, name)` otherwise (fail-loud on a
  missing field), via the generic callable path in `_eval_node` (no change needed
  there). `Input` is unchanged: a bare-tensor input still *is* the value (all
  current single-input graphs + demos untouched); a struct input is read by
  `field(...)` nodes. Done; see `multi_input_demo.py`.
- `_SaveSession` / `Graph.forward`: accept either a tensor (seed `{Input: x}`, as
  now) or a batch struct (seed `{Input: batch}`, fields read lazily). No change to
  the memoized-pass machinery.
- `sae.py` `NormalizedModel`: add a `target` notion. `x = field("input")`,
  `y = field("target")` with `target` defaulting to `input` when absent (preserves
  today's `y_norm = x_norm` for autoencoders; enables `!=` for transcoders).
- **`total_loss` handle (resurrect D-04, cleanly).** Each loss term reads its
  weight as a field: `total = Σ field(f"w_{k}") * term_k`. `loss(batch, coeffs=…)`
  packs `coeffs` into the batch as `w_*` fields; `save(total_loss)` is the backprop
  loss. No buffers ⇒ no `.to()` staleness (the bug that killed D-04). The unweighted
  terms are still saved for logging.

## Migration / back-compat

- Purely additive. `Input`-as-tensor stays the default; the 5 demos and the model
  specs (`GatedModelSpec` etc.) keep using `Input` and don't change.
- The trainable opts in: build the batch struct from `(x, y, coeffs)`, run one
  `forward_values(batch)`, `save(total_loss)`. `GatedModelSpec`/`GatedAuxSpec`
  need no change (they read `x`); only `NormalizedModel` + `GraphTrainable.loss`
  move onto fields.
- Validate: all three test suites + the 5 demos + GPU through the real Trainer
  (must train identically), exactly as the normalizer change was validated.

## Open questions for Glen

1. **DictBatch type.** Reuse `SAETrainBatch` directly as the seeded struct, or a
   thin cgraph `Batch` (dict-backed) that `SAETrainBatch` adapts into? Leaning
   thin-`Batch` so cgraph stays decoupled from the saeco batch type, with an
   adapter at the trainable boundary.
2. **Weights as fields vs. a single weight-vector field.** Per-loss `w_{k}` fields
   (simple, readable) vs. one `weights` vector indexed by loss order (fewer nodes,
   but reintroduces positional coupling). Leaning per-loss fields.
3. **Missing-field policy.** `field("target")` when the batch has no `target`:
   fall back to `field("input")` (convenient, what autoencoders want) vs. hard
   error (fail-loud, principle 4). Leaning *explicit default at the call site*
   (`NormalizedModel` wires `target or input`), so `field` itself stays fail-loud.

If this looks right I'll implement B-on-A behind the existing single-input path,
land the `total_loss` handle first (smallest, highest-value slice), then the
transcoder `target`. Each its own commit, each validated GPU-through-Trainer.

---

# Update after Glen's review — typed contracts, the converter fork, multi-output

Glen reviewed and asked for **typed `DictBatch` contracts, no string accesses**, an
**override escape hatch**, **converters**, and raised a deeper reframe (converter
on the front vs. batch-unpacking baked into the graph; and the symmetric
multi-output/logging case). His central worry: *is the typed surface expressible?*

## Feasibility: CONFIRMED (with evidence) — and BUILT

`field(Batch.input)` type-checks. A `_BatchMeta` metaclass makes each class-level
field a typed `FieldRef`; **pyright reports `Batch.input: Tensor`, `Batch.weight:
float`** (probed directly), and `field(Any) -> ValueRef` accepts it. `DictBatch` is
`dataclass_transform`'d, so `Batch(input=…, target=…, weight=…)` is *typed
construction* — pyright catches wrong field types and missing fields. A
constructed instance exposes values as instance attributes (`b.input` is the
tensor); a declared-but-unprovided field fails loud on access (a `__getattribute__`
guard catches the would-be class-`FieldRef` fallback). All shipped (commit after
this doc): `dictbatch.py`, `field()` duck-typing `str | FieldRef`, the rewritten
typed `multi_input_demo.py`, a unit test — `dictbatch.py` + the demo are
**pyright-clean on the user surface**. This realizes Glen's exact sketch.

**Override** (his "set y and w, pass z=Input"): no new mechanism — declare a field
as a slot defaulting to the batch read, `x: Arg = arg(default=field(Batch.input))`,
and bind `x` to a plain value at construction to bypass the batch. Demoed + tested.

## The converter fork dissolves — it's function composition into a typed contract

Glen's tension was: does the spec *bake in* `Batch -> x,y,w`, or declare abstract
inputs `x,y,w` with a *converter* on the front? Tested empirically (`/tmp` probe):
**a graph reads its own typed input-contract `DictBatch`, and a converter is just a
typed function `Source -> InputContract`.** No new graph machinery:

    class AEInputs(DictBatch):          # the graph's INPUT CONTRACT
        x: torch.Tensor
        target: torch.Tensor
    class Recon(GraphSpec):
        xn = field(AEInputs.x); y = field(AEInputs.target)
        output = op(_l2)(op(torch.tanh)(xn), y)

    def autoencoder(src) -> AEInputs: return AEInputs(x=src, target=src)   # y == x
    def transcoder(a, b) -> AEInputs: return AEInputs(x=a, target=b)       # y != x
    g.forward(autoencoder(raw))   # one graph...
    g.forward(transcoder(a, b))   # ...two converters, both typed, verified

So "baked-in vs converter-front" is a false dichotomy: the graph names *its own
contract's* fields (its interface), and converters typed `Source -> Contract` adapt
any external data to it — reusability **and** full typing. The graph is coupled
only to its own input contract, which is exactly what an interface should be. This
answers the transcoder case (`y_norm != x_norm`) too: it's a different converter,
not a different graph. (Resolves open-question #3 — missing-`target` is the
converter's job, so `field` stays fail-loud.)

## Multi-output: the dual (and where Glen's D-03 note lands)

Glen: outputs are the mirror image — "a recipe for taking a forward pass and
turning it into a DictBatch; the mapping between fields and which valuerefs' outputs
go into them," and *logging is one such recipe built dynamically by bubble-up,
executed inside a `forward_values` block.* And his D-03 note: bubble-up matters
precisely because we log metrics we **can't plan from outside**, so we need a
**contract for passing what-to-log up from the inside**.

These are the same thing. Proposed design (the symmetric half):

- An **output recipe** is a typed map `{field_name -> ValueRef}` targeting an output
  `DictBatch`. The statically-known outputs (e.g. `recon`, `acts`, `loss`) are a
  typed `DictBatch`; running the graph fills it. This is `GraphSpec.of(**handles)`
  generalized to *produce a typed batch* instead of a bare graph.
- **Logging is a dynamically-built output recipe.** Bubble-up (`collect`, already
  built — D-03/D-05) walks the graph and gathers each site's `(name, handle)`; that
  set *is* the recipe; executing it in the `forward_values` block writes the logged
  `DictBatch` / cache. The fields aren't known from outside — hence Glen's
  `@DictBatch.auto_other_fields`: the logging batch has its known fields **plus**
  whatever bubbled up. So `auto_other_fields` is exactly the typed-core +
  dynamic-tail contract the logging case needs.
- This subsumes today's `GraphTrainable.loss` cache-writing: instead of hand-listing
  `cache.L0 = save(...)`, the trainable runs the bubbled-up output recipe, and new
  inside-metrics appear in the log without the outside naming them — which is the
  whole point of D-03.

**Type-system caveat (the honest part):** the *static* outputs type cleanly (a
`DictBatch` of handles). The *dynamic* bubbled-up tail can't be statically typed
(discovered at runtime) — `auto_other_fields` is deliberately the untyped escape
valve. That's the right place to give up static typing, since "metrics you can't
plan from outside" are by definition not statically known.

## Revised plan (smallest valuable slices, each its own commit, GPU-validated)

1. **DONE** — typed `DictBatch` + `field(Batch.x)` + override + converter framing.
2. **`total_loss` handle** — the trainable builds a typed input `DictBatch`
   (`input`, `target`, per-loss `w_*`), reads weights as fields, exposes
   `total = Σ field(w_k)·term_k` as a handle. Device-clean (fields are extraction
   nodes, not the stale buffers that killed D-04). Smallest migration; biggest win.
3. **Transcoder `target`** — `NormalizedModel` reads `field(Inputs.target)`;
   autoencoder converter sets `target=input`. Unlocks `y_norm != x_norm`.
4. **Output recipe + bubble-up logging** — generalize `GraphSpec.of` to produce a
   typed output `DictBatch`; the trainable's cache-write becomes a bubbled-up
   output recipe (the D-03 contract). Largest; do last, likely its own design pass.

## Decisions still genuinely yours

- **Input-contract location.** A cgraph-side `DictBatch` the trainable builds and
  `SAETrainBatch` adapts into (keeps cgraph decoupled — my lean), vs. having
  `SAETrainBatch` *be* a `DictBatch`. The converter framing makes the adapter trivial.
- **Weights:** per-loss `w_{k}` fields (readable — my lean) vs. one weight vector
  field (fewer nodes, positional). With `auto_other_fields` the per-loss fields cost
  nothing structurally.
- **How far to push output recipes now** — just the `total_loss` slice (#2), or the
  full bubble-up logging contract (#4) in this pass? #4 reshapes the cache path, so
  I'd want your nod on its shape before building.

---

# Refinement 2 (Glen) — contracts live only at the I/O boundary; inside it is ref-piping

A sharpening of the converter/contract picture. The realization: **internal
composition needs no contracts at all.** When a subgraph is embedded in an outer
graph, you do not bundle its inputs into a DictBatch and you do not give it an
output type — you just **pipe refs**: bind the subgraph's input slots to the outer
graph's refs, and the outer graph reads whichever of the subgraph's handles it
wants. That is exactly what `arg()` + `bind` + `sub_slot` already do.

So the model is:

- **A graph's inputs are `arg()` slots.** Binding a slot is the universal input
  mechanism. You bind it to: an outer graph's ref (composition), a `field(Batch.x)`
  read (a real batch at the boundary — D-09), or a constant (override — D-09).
  These are all the same `bind`; `field`/`DictBatch` is just boundary sugar.
- **An "input contract" (when you want a *named* multi-input interface) is just a
  simple spec of only `arg()`s** — an interface other graphs read and callers bind.
  For the trivial case the "contract" is just a bare tensor `Input`.
- **A subgraph needs NO output contract / return type.** What matters is which of
  its nodes the outer graph reads (its handles stay navigable — D-02). Output
  *contracts* (DictBatch) matter only at the true outer boundary, where you coerce
  the result into a typed batch for an external consumer / the log (`produce(spec,
  Contract, batch) -> Contract`, point-3 design).

Net: `DictBatch` contracts are a *boundary* concern (real batch in, typed log out),
not a *composition* concern. Inside, it is refs binding to slots — already built.
The remaining new surface is small: the boundary coercion (`produce`) + the
input-side compose (`with_inputs`, which is just binding outer refs to inner slots
by name), both on `GraphSpec`.

(Parked, see STACK.md: check subgraph node-name duplication in the name maps.)
