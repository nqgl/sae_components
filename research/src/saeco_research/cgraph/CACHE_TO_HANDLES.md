# Replacing the cache with output-handles (assessment)

The cache is a **stringly-keyed side channel for reaching a value out of the
middle of a forward pass** — penalties, metrics, intermediate activations,
normalizer factors. The graph model already makes every intermediate a typed
`ValueRef`; exposing a graph's named intermediates as **handles** turns "reach
into the middle" into "grab a typed ref", and `ref.build()` runs only that ref's
cone. Validated in `handles_demo.py`: an `SAESpec` exposes `acts`/`recon`;
`sae.produce(sae.acts)` runs only the encoder; splicing `sae.acts` into a penalty
fuses (encoder once).

## What the cache is used for today → the handle equivalent

| Cache use (today) | How it works now | Handle equivalent |
|---|---|---|
| **Sparsity penalty** | a penalty `PassThrough` writes `cache.sparsity_penalty`; `SparsityPenaltyLoss` reads it via `cache._ancestor.search(...)` | model exposes `acts`; the sparsity loss is `L1(sae.acts)` — a value node spliced into the loss graph. **Done** (`handles_demo`, `gated_graph`). |
| **Metrics** (`ActMetrics` L0/L1) | metric modules write to a globalized cache; logged | metrics are nodes on the `acts` handle; log by reading those nodes' outputs (see "logging"). |
| **Intermediate activations** (preacts, gate, acts) read by losses/resampler | written to subcaches during forward, found later | named handles (`sae.acts`, `sae.gate`, ...) referenced directly. |
| **Normalizer** scale/shift | `forward` writes `cache.scale/shift`; `invert` reads them | the `apply`/`invert` ref bundle — scale/shift are refs reused on both sides. **Done** (`demo.py`). |
| **ReuseForward** `forward_reuse_dict` | root-cache memo dedups shared computation across calls | hash-consing on a shared builder. **Done** (fusion). |
| **Resampling** reads activations + freqs | resampler pulls from cache during/after forward | reference the `acts` handle; a freq-tracker is a node consuming it. **Open.** |
| **Logging** (`logdict` walks cache) | stored watched values are itemized | mark handles to log (a `to_log` set of refs); read their outputs after a run. **Open.** |

## What this needs to be a full replacement (open)

1. **A logging story** — today `logdict` reads everything watched. With handles:
   designate which refs to log, run a graph whose output bundles them (or read
   them off a single run), and itemize. Cheap, but needs a defined surface.
2. **Multi-output / `DictBatch` returns** — many handles at once. Glen's idea: a
   `DictBatch` return type by default; a graph "run" can target a *set* of
   handles (one toposort over their union → still computes each once). The
   single-target `ref.build()` is the special case.
3. **Resampler integration** — the resampler walks `named_modules` and reads
   activations; rephrase it to consume the `acts` handle (and `produce(acts)` to
   get encode-only activations cheaply).
4. **Predictable handle interfaces** — an `SAESpec(GraphSpec)` (and friends) with
   a *contract* for which handles exist (`acts`, `recon`, ...), so losses/metrics/
   resamplers can rely on them by name+type rather than by cache string.
5. **Impurity / side effects** — anything that *must* run for effect (e.g. an EMA
   tracker update) isn't pulled in by a pure-value toposort. Needs an explicit
   "effect" edge or to be a consumed value node.

## Mechanism in place (this round)

- `GraphSpec` instances expose **every named body value as a handle** (via
  `__getattr__` over the inlined node-map); `inline` now returns the full
  template-node → target-ref map.
- `ValueRef.build()` / `spec.produce(ref)` freeze a Graph targeting one handle —
  it computes only that handle's cone.
- Handles are refs in the construction builder, so splicing one into another
  graph **fuses** (hash-consing).
