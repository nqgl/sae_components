# cgraph2 — architecture overview

The shape of the system as of the overnight run. Read INDEX.md first; DECISIONS.md
for the *why* of each call.

## One sentence

A sparse autoencoder's whole training computation — model, normalizer, losses,
metrics — is built as **one graph of interned nodes**, composed by plugging graphs
into each other's slots and applying them, run in a single fused pass.

## The layers

```
core.py    interned nodes + execution
spec.py    GraphSpec: the authoring surface AND the runnable graph
sae.py     graph-native SAE / Normalizer / Loss / Trainable + bubble-up sites
architectures/gated_graph/  the gated SAE, assembled like stock saeco
```

### core — interned relational DAG

- A **`ValueRef`** is an immutable node `(producer, inputs)`, **interned by
  structure** in a global table. Two structurally-identical sub-DAGs are the same
  object, so fusion/dedup is automatic and global. Producers: `nn.Module`s, params/
  buffers (value sources), `Op`s (stateless fns, consed by *structure* via
  `funcstd.standardize`, never by label), and the single global `Input`.
- **`bind(target, subs)`** re-interns a cone substituting producers/value-nodes —
  the one primitive behind slot binding, normalization, and `at`.
- **`at(handle, x)`** = `bind(handle, {Input: x})` — "apply this sub-graph to a new
  input." A producer may define `__graph_rebind__` to control its re-interning
  (how a sub-model slot splices in).
- **Running** is a single fused, memoized pass: `_SaveSession` (`forward_values`)
  evaluates the *union* of requested handles' cones, each interned node at most
  once, and supports `save.after(h, fn)` deferred side-effects.
- **`collect(targets, SiteType)`** is the graph-native `Module.apply`: walk the
  cone and bubble up nodes whose producer is a site.
- **`field(name)`** reads a named field off a *structured* input seeded at `Input`
  (the multi-input primitive, D-08): a graph reads either `Input` as the value
  (single-input, every current graph) or `field("x")`/`field("target")` off a
  struct. Additive; the migration onto it (the `total_loss` handle, transcoders)
  is proposed in `docs/DESIGN-multi-input.md`.

### spec — GraphSpec is the graph

A `GraphSpec` subclass declares slots as typed fields and wires handles in the
body; the instance is the graph (D-01):

- **Slots:** `op_slot()` (a producer), `arg()` (a value/input), `sub_slot()` (a
  whole sub-graph). Handles are body assignments; `output` is the default.
- **Typed handles:** `g.acts` is a `ValueRef` to the type checker; a typo errors.
- **Composition:** `g(x)` / `g.apply_to(x)` applies a graph to a new input;
  `m(x).acts` plugs a sub-model in (sub_slot) and reads its handle. Bound
  sub-graphs are kept as **navigable children** — structure is preserved, not
  flattened (D-02).
- **Inheritance:** subclasses inherit slots/handles and may override a slot's
  default (`class L2Loss(ReconLoss): loss_fn = op_slot(default=_mse)`); an
  output-less base is abstract.
- **Runnable:** `g.forward_values` / `g.collect` / `g.forward`; `g.register_into(
  module)` registers leaf producers onto an owning `nn.Module`. `GraphSpec.of(
  **handles)` builds one dynamically for runtime-sized aggregates.
- **Views:** `g.describe()` shows the nested-subgraph structure (the navigable
  children, D-02); `g.render()` shows the flat **deduped DAG** — every node printed
  once, so a shared node (the encoder, the normalize op) appears a single time.
  The structure view and the execution view, side by side.

### sae — the SAE, graph-native

The normalization win: instead of saeco's `normalizer[loss[model]]`, we get
`loss[normalizer[model]]`.

- **`SAEGraphSpec`** — base for model graphs (define `acts`/`recon`/`output`).
- **`AffineNormalizer`** — a clean static invertible affine normalizer that
  consumes a `GNConfig` and is the *static collapse* of
  `StaticInvertibleGeneralizedNormalizer` (folds the four sequential factors into
  two primed buffers `mu`/`scale`). **Codex-verified value-identical across all 16
  `{DONTUSE,PRIMED}` configs** (D-07); dynamic aggs raise. `GraphNormalizer`
  composes it into graphs (`normalize` node + a stably-consed `denorm` op).
- **`NormalizedModel(GraphSpec)`** — `normalizer[model]`: `model(normalize(Input))`,
  exposing `y_norm`/`y_pred_norm`/`acts` and a denormalized `out`.
- **Losses are GraphSpecs** over a `NormalizedModel` sub_slot
  (`GraphL2Loss`/`GraphCosimLoss`/`GraphSparsityLoss`); `@loss_prop` returns
  `GraphL2Loss.on(model)` (a `LossRef`).
- **Bubble-up sites:** `ActMetrics` (a GraphSpec) bubbles up L0/L1 via a
  `_MetricsSite` marker (`metered(acts)`); `FreqTracker` tracks an EMA with a
  *deferred* update onto the loss pass (D-03/D-05).
- **`GraphTrainable(Trainable)`** — init mirrors stock `Trainable(models,
  resampler, losses, normalizer)`. It wraps each model in a `NormalizedModel`,
  resolves losses, builds the whole computation as one `GraphSpec.of` graph,
  registers it, and `loss()` runs one `forward_values` pass writing the cache the
  trainer reads. The loss terms share that one fused pass; the (dynamic) coeff
  weighting is applied **in Python** — `total = Σ term·coeff` — because a graph
  weight source-node would capture a buffer that `.to(device)` orphans (D-04 was
  superseded by this device-safe form; a true `total_loss` *handle* awaits weights
  as graph **inputs** via `field()`, see below).

### the arch — like stock saeco

`GatedGraphSAE(Architecture[GatedGraphConfig])` is the stock gated arch
part-for-part — `setup()` wrappers, `@model_prop`/`@aux_model_prop`/`@loss_prop` —
but the parts are graphs. The shared encoder is **one interned node** across the
mag path, gate path, aux model, and sparsity loss: it runs once, no
`ReuseForward`. It trains end-to-end through the real `Trainer` on cached gemma-4
data (L2 0.6→0.02, cosim 0.76→0.99, L0-targeting active).

## What's deferred (see TODO.md / DECISIONS.md)

- **Multi-input migration.** The `field()` primitive is built (D-08); the migration
  onto it — `NormalizedModel` reading a `target` field (transcoder `y_norm !=
  x_norm`) and the trainable exposing a real `total_loss` handle (loss weights as
  graph inputs, resurrecting D-04 device-clean) — is proposed and awaits greenlight
  (`docs/DESIGN-multi-input.md`, open questions at the bottom).
- **Resampling reset path.** The freq-tracker bubble-up + deferred EMA update is
  built and *live* in the gated arch (a `FreqTracker` is spliced and `below_*` freq
  metrics are logged); the resampler's dead-feature *reset* path is not yet wired.
- **Retire cgraph v1** (`../../cgraph/`) once nothing references it.
