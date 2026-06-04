# Design proposal — multi-input graphs (DictBatch) & the `total_loss` handle

**Status:** the **`field()` primitive is built, tested, demoed** (D-08); the
**migration is the proposal awaiting greenlight.** Written + the primitive landed
during the overnight run because this is the biggest repeatedly-deferred item and
it gates two things we keep wanting (a true `total_loss` handle, transcoders).

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
