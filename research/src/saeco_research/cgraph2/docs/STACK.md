# cgraph2 — stack (LIFO "don't forget this" scratch)

Most-recent on top. Pop (strike through or delete) when addressed. These are
parked questions/tasks we explicitly don't want to lose, not the active plan
(that's PROGRESS.md / TODO.md).

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
