# cgraph2 docs — index & how to use them

Central plan for the cgraph2 documentation. Read this first.

## The docs

- **INDEX.md** (this file) — what each doc is for. Keep the active-docs list current.
- **ARCHITECTURE.md** — the shape of the system (layers, key types, what's deferred).
  The doc to read for the big picture.
- **DECISIONS.md** — the design-decisions log. Every significant design call gets
  an entry: the question, the options, the choice, and *why*. Append-only;
  supersede rather than rewrite (note "superseded by D-NN").
- **PROGRESS.md** — chronological work log + the **resume protocol** for autonomous
  runs. Each work session appends what it did, what's green, what's next. This is
  the doc to read to pick up where things left off.
- **../TODO.md** — short forward-looking task list (normalizer/GNConfig, etc.).
- **../glen-inbox.md** — Glen's thoughts dump (his; don't rewrite, only append a
  `> resolution` note when an item is addressed).

## Conventions

- A *significant* decision (changes the model, the API surface, or the execution
  semantics) → an entry in DECISIONS.md. A routine choice → just a code comment.
- When in doubt about a design call during autonomous work: pick the option that
  best fits the principles below, **record it in DECISIONS.md with the reasoning
  and the alternative**, and move on. Don't block.
- Stale/superseded material: don't delete history; mark it superseded and point
  forward. v1 (`../../cgraph/`: TASKS.md, DONE.md, PROD_READINESS.md,
  CACHE_TO_HANDLES.md) is **historical** — the relational v2 here supersedes it;
  consult it only for rationale.

## Principles (extrapolated from the work so far)

1. **Everything is a graph.** Model, normalizer, losses, metrics — all
   `GraphSpec`s, composed by plugging one into another's `sub_slot` and applying
   it (`g(x).handle`). A graph is the runnable context; a bare ref means something
   only relative to its graph.
2. **Keep structure; dedup at execution.** Nested subgraphs keep their boundary
   (navigable), and the deduped execution DAG is derived from the requested
   outputs (interning gives the fusion). Don't flatten away the structure.
3. **Bubble up.** Information (metrics, freqs, logs) is produced wherever it lives
   and pulled from outside by walking the graph — no middle layer needs to know
   where it is. Stateful side-effects (freq update) defer onto the forward pass.
4. **Fail loud at the boundary; type the seams.** Slots/handles are type-visible;
   a typo errors. Loud errors over silent degradation.
5. **Small elegant interface over intricate machinery.** Hide the interning /
   registration / dedup; expose `g.handle`, `g.forward_values`, `g.collect`.
6. **Match the codebase.** attrs `@define` + factory classmethods for plain data;
   `cl.Module` for torch; mirror stock saeco's Architecture/Trainable flow.
