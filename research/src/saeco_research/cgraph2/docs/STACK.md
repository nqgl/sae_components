# cgraph2 — stack (LIFO "don't forget this" scratch)

Most-recent on top. Pop (strike through or delete) when addressed. These are
parked questions/tasks we explicitly don't want to lose, not the active plan
(that's PROGRESS.md / TODO.md).

---

## ▸ NormalizedModel[T] generic — typed model-prop returns  (pushed 2026-06-04, Glen)

(uncertain if good) Make `NormalizedModel` generic — `NormalizedModel[T]` — so a
`@model_prop` could return `NormalizedModel[T]` rather than `T`. I.e. parametrize
the wrapper by the model graph type it wraps, so the typed handles flow through the
normalization wrapper. Explore whether this types cleanly (the wrapper exposes
`acts`/`y_norm`/… ; parametrizing by the inner model's contract may or may not be
expressible) and whether it's worth the generic machinery.

---

## ▸ subgraph node-name duplication  (pushed 2026-06-04, Glen)

When subgraphs get composed, check what happens if they cause **node-name
duplication** — i.e. how nodes end up getting named and referred to (the
`core._names` / `GraphSpec.names` / `render()` / `_SaveSession.save` name maps;
`_names` falls back to `f"n{i}"` by toposort index, so two different cones could
both produce an `n3`). Is that ever load-bearing (cache subkeys, collisions in a
shared name map), or purely cosmetic? Not urgent — just don't forget to look.
Parked during the input-contract / ref-piping design discussion.
