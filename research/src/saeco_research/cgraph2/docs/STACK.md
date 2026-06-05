# cgraph2 — stack (LIFO "don't forget this" scratch)

Most-recent on top. Pop (strike through or delete) when addressed. These are
parked questions/tasks we explicitly don't want to lose, not the active plan
(that's PROGRESS.md / TODO.md).

---

## ▸ pass a config to a GraphSpec  (pushed 2026-06-05, Glen)

Sometimes a GraphSpec needs a config (e.g. the normalizer's GNConfig, thresholds,
flags) — figure out how a config flows into a class-block GraphSpec. Options: a
plain `__init__`-style arg the metaclass threads through; a `config` slot; or pass
it to the producers (modules) before binding. Decide the idiom.

---

## ▸ do we need an `output` handle?  (pushed 2026-06-05, Glen)

Re-examine whether every GraphSpec must declare `output` (the default `forward`
target / what makes it non-abstract). For a spec used only for its named handles
(a model read via `.acts`/`.recon`, a converter read by field name, an interface
provider), a single `output` is arbitrary. Could non-abstract-ness come from
"has ≥1 handle" instead, with `forward` requiring an explicit target?

---

## ~~ValueRef[T] generic over producer~~  DONE (2026-06-05, commit 69f1877)

`ValueRef[T]` (producer: T); `ref[T] -> ValueRef[T]`; `collect[S](…, type[S]) ->
list[ValueRef[S]]` + `GraphSpec.collect[S]`. `collect(FreqTracker)` is now
`list[ValueRef[FreqTracker]]` and `site.producer` is a `FreqTracker`
(pyright-confirmed); `_freq_sites` typed.

---

## ~~NormalizedModel[T] generic~~  DONE (2026-06-05)

`NormalizedModel[T: SAEModel]` (`model: T`); `GraphNormalizer.wrap[T](model: T) ->
NormalizedModel[T]`. Declared `acts`/`recon` on `SAEGraphSpec` so it statically
satisfies `SAEModel` (annotation-only; metaclass ignores bare annotations). Ready
for a `@model_prop` to return `NormalizedModel[T]` if/when we want it (deferred --
it's an arch-API change).

---

## ▸ subgraph node-name duplication  (pushed 2026-06-04, Glen)

When subgraphs get composed, check what happens if they cause **node-name
duplication** — i.e. how nodes end up getting named and referred to (the
`core._names` / `GraphSpec.names` / `render()` / `_SaveSession.save` name maps;
`_names` falls back to `f"n{i}"` by toposort index, so two different cones could
both produce an `n3`). Is that ever load-bearing (cache subkeys, collisions in a
shared name map), or purely cosmetic? Not urgent — just don't forget to look.
Parked during the input-contract / ref-piping design discussion.
