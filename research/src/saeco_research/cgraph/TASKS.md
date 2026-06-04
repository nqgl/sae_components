# cgraph — active tasks

Prototype of a DAG composition primitive for `saeco.core` that subsumes
`cl.ReuseForward` (shared nodes are just nodes with two out-edges) and the
normalizer's shared-cache write-back. Started 2026-06-01.

Completed work is logged in [DONE.md](./DONE.md). Corners cut on the way to a
working prototype — and what production needs instead — are tracked in
[PROD_READINESS.md](./PROD_READINESS.md). **This is on a path to production**, so
prefer solid designs; when a corner is cut to keep moving, log it there.

## Design decisions settled (don't relitigate without reason)

- **Executor dedups by node identity.** A node referenced by N consumers runs
  once. This is what removes the need for `ReuseForward`.
- **Operators on `Ref` mean arithmetic** (`*`, `+`, `-`, `/` → fan-in op nodes),
  not structural composition. Structure is `g(producer, *inputs)`. (Combinator
  operators like `>>`/`|` can express sharing via variable binding too, but we
  reserve operators for math since SAE code is arithmetic-heavy.)
- **Naming via class-namespace + decorator, not descriptors.** `@g.namespace`
  inspects `vars(cls)` to learn node names from the variable each ref was bound
  to. Refs stay plain value objects; naming is non-load-bearing (no class →
  auto `n#` names, nothing breaks). Walrus binds into the class namespace too,
  so terse inline naming works.
- **Graph-native modules expose `apply(g, x) -> bundle` + the bundle owns
  `invert`**, instead of one opaque `forward`. This is what keeps the normalizer
  contained while sharing scale/shift across normalize+invert. Leaf modules stay
  `forward`-based (option (a)); only composites grow `apply`.
- **A built `Graph` is a `cl.Module`**, so it nests as a node in a parent graph.
- **Fusion = hash-consing at construction on a shared builder.** `g(producer,
  *inputs)` returns an existing equivalent node, so independently-authored
  subgraphs that share a builder + upstream refs fuse automatically (no pass, no
  cross-awareness). Architecture should give all parts (model/aux/losses/norm)
  ONE builder and have them *emit refs* — not freeze separate Graphs early.
  `fresh=True` opts impure producers out. Merging pre-frozen graphs (inline+CSE)
  is the fragile fallback we want to avoid needing.
- **Finalization: `g.output = ref` inside the `@g.namespace` body; the decorator
  builds and returns the `Graph`.** `g.build(ref)` is the explicit non-namespace
  path. (Caveat: the decorator-returns-Graph form isn't type-visible to pyright —
  see task #6 / slot model.)
- **Style: attrs `@define` + factory classmethods** for plain classes (matches
  the repo). nn.Module subclasses stay torch modules.
- **Slots > separate inputs/params (planned, #6).** Inputs and parameters are one
  concept — slots; an input is unbound, a parameter is bound, binding is partial
  application. The graph's call signature is its unbound slots.

## Active tasks

- [x] **#1 — mock-Trainable wiring.** Done — see [DONE.md](./DONE.md). Confirmed
  the seam, the penalty/loss cache integration (`search` works through the graph
  executor), and that it trains. Surfaced the loss-layer ReuseForward dependency
  → new task #4.
- [ ] **#2 — naming polish.** Nested namespaces so emitter-internal nodes
  (normalizer's shift/scale) and subgraph nodes get readable dotted names
  (`norm.scale`). Add `to_dot()` for real diagrams.
- [~] **#3 — real-arch rewrite.** FIRST INCREMENT DONE (`gated_graph.py`): the
  gated arch (main + aux + 3 losses) on the slot model, paralleling saeco
  (`GraphLoss` ~ `Loss`, `GraphTrainable(Trainable)`, `GatedGraph` ~ `Gated`);
  enc_gate fused once across main+aux+sparsity; trains. Surfaced+fixed the dedup
  issue (nn.ReLU instances -> torch.relu). **Remaining:** real `Initializer`
  modules (LinEncoder/decoder, tied weights, wrappers NormFeatures/Orthogonalize),
  the real `Trainable` contract (metrics, resampler, coeffs from `train_cfg`,
  normalization placement), descriptor integration (`@model_prop`/`@loss_prop`
  returning graphs — types can be loose per Glen), and loss-space decision.
- [~] **#4 — total loss as one fused graph (from #1).** DESIGN VALIDATED +
  fusion mechanism BUILT. `megaloss_spike.py`: main + aux + penalty + per-loss
  math + weighted sum as ONE DAG; penalty = value edge (retires `search`);
  detach = explicit edge. `fusion_spike.py` + hash-consing in `core.py`:
  independently-authored subgraphs sharing a builder fuse automatically
  (encoder once), `fresh=True` opts impure producers out. **Remaining (real
  implementation):** (1) the **emitter-descriptor wiring** — how do
  `@model_prop`/`@aux_model_prop`/`@loss_prop` share one builder (framework
  threads a `g`, or descriptors return `(g,x)->ref` emitters)?; (2) migrate real
  `Loss` off model-re-run + `search` to consuming value nodes; (3) purity policy
  for impure producers; (4) coeffs/metrics/logging. See PROD_READINESS #10/#13/#14/#15.
- [ ] **#5 — collision-safe node-output cache writes.** Node-output `_write`s are
  unwatched/harmless but could hijack a watched field if a node is named e.g.
  `L1`/`loss`. Namespace them or guard against watched names. PROD_READINESS #3.
- [~] **#6 — slot model (foundational).** EXPLICIT API + `dataclass_transform`
  NAMESPACE + thread-`T` DONE & validated (`slots.py`, `namespace.py`,
  `slots_demo.py`, `namespace_demo.py`, `fusion_stress.py`). Type-visible
  (pyright reveals/enforces both the explicit `(g, **slots)` signature and the
  class-namespace construction signature), fuses through nesting (shared
  builder / template inlining). Frozen-subgraph fusion via `inline=True`; slot
  defaults (`op_slot(default=...)` AND concrete `compare: OpRef = g(nn.ReLU())`
  via annotation-based detection); unified `g(...)`: all DONE. **Remaining:**
  (a) the *bare-annotation* slot form only — `x: ValueRef` (no `= arg()`) and
  `compare: OpRef = nn.ReLU()` (bare module, not wrapped in `g(...)`) — via a
  `__missing__`/`__setitem__` namespace; the `= g(...)` and `= arg()`/`op_slot()`
  forms already work;
  (b) multi value-input — **decided: support a `DictBatch` input type (explicitly
  annotated) rather than multiple input slots**; not yet built (do later);
  (c) `Op[Args, T]` arity-typed producer slots; (d) decide whether to auto-inline
  cgraph Graphs by default (fuse-always vs keep boundary); (e) benign pyright
  notes on metaclass-internal `_template`/`_out`/`.default` (runtime-fine). Then
  the `src/saeco` rewrite (#3) on top. Original notes below.

  Unify inputs and parameters as *slots*:
  an input is an unbound slot, a parameter is a statically-bound one, binding is
  `functools.partial`. A graph's call signature = its unbound slots (names +
  order, inputs-at-top). Make it type-visible via `dataclass_transform` on the
  namespace decorator — slot declared by `x = g.input` (sentinel marks
  required), annotations feed the synthesized signature. **Motivation/urgency:**
  the current `@g.namespace`-returns-`Graph` pattern is runtime-clean but pyright
  treats the decorated name as the *class*, not a `Graph` — so it's not
  type-visible. The slot model + `dataclass_transform` is the path that type
  checkers honor. Validate in the prototype BEFORE the `src/saeco` rewrite (#3),
  since the rewrite should stand on the slot API. Known wrinkle to solve:
  annotation-only names aren't usable in the class body (need the `= g.input`
  sentinel). The `Arg`/callable-template idea folds in here (a template = a graph
  with unbound slots).

## Rough edges (observed in the prototype)

- **Emitter-internal nodes auto-name as `n#`.** Naming only harvests class-attr
  bindings, and `apply` creates its nodes internally, so they don't get pretty
  names — the producer label (`MeanShift`, `StdScale`) carries the meaning
  instead. Fixed by #2 (nested namespaces): `apply` running inside its own named
  sub-namespace would yield `norm.shift`, `norm.scale`.
- **Nested graph prints as bare `Graph`** — should show its output name or label.
- **`.split()` and `.detach()` exist on `Ref` but aren't exercised** by the two
  demos yet. Need coverage.
- **pyright flags the `cgraph` import as unknown** — editable install resolves
  `saeco_research` to the `fp_info` worktree, not this one; runtime `PYTHONPATH`
  shadows it. Not a real error; pyright isn't a gate.

## Big next: relational / frozen-graph redesign (Glen's direction)

- [~] **#7 — pure relational graphs + global Input** (replace the mutable
  `GraphDef` builder). FIRST INCREMENT DONE — clean v2 in `cgraph2/` (interned
  `ValueRef`, global `Input`, `GraphSpec`-only, structural fusion, `produce`,
  handles); `cgraph2/demo.py` green. **Remaining:** intern-table lifetime/scoping;
  multi-input via `DictBatch`; port gated arch + GraphLoss + GraphTrainable onto
  v2; logging/resampler handles; then retire v1. A `Node` becomes an immutable value `(producer, inputs)`
  interned by structure (cons key = producer identity + input-node ids — the
  function-identity fix is the prerequisite). `Input` is a single global marker
  node, so every graph that reads input shares it → composition fuses
  *structurally*, no inline/re-emit. A `ValueRef` is then just a `Node` (no
  builder ref); `Arg(default=Input)` lets `Nest()` drop the `x=g.input()`.
  Subsumes: inline/re-emit (composition shares nodes), the mutable builder
  (interning is global/contextual), and `x=g.input()`. GraphSpec stays the only
  authoring surface (Glen is OK with that). Open: intern-table scoping/lifetime;
  slot binding as structural substitution; how `produce(target)` + multi-target
  runs evaluate. This is the elegant form the current `GraphDef`/inline machinery
  is approximating.

## Nits

- [x] **`Op` label optional** — `Op(torch.exp)` auto-labels "exp"; module labels
  use `nn.Module._get_name()`. Done.

## Open design questions

- **`Arg` placeholder / callable node-templates.** Idea: a sentinel `Arg` that
  marks holes in `g(producer, *args)`, turning the result into a *callable
  template* that fills the holes when called. Would eliminate `Op`, give
  Keras-functional ergonomics, enable tied-weight reuse, and allow positional
  control. Sketches floated:
  ```python
  exp = g(torch.exp, Arg)          # template; exp(x) -> torch.exp(x)
  relu = g(nn.ReLU())              # no args == single implicit Arg
  mag  = relu(h * expr + g(bias))  # apply the template
  named_param = g(param)           # 0 holes, value producer -> source node
  f = g(fn_of_two, Arg, param)     # Arg controls where data flows: fn(x, param)
  ```
  Tensions to resolve: (a) `g(...)` would return `Ref` *or* `Template` —
  type-checker story (overloads can't see "contains Arg"); maybe `Ref` is always
  returned and is callable iff it has unfilled holes. (b) Template nodes must not
  register in the graph until applied. (c) Disambiguate no-arg value producers
  (`g(param)` → source node) from no-arg function producers (`g(relu)` →
  template). Promising surface sugar; layer on top of the core, don't block #1.
  Decide after a small spike.
- **Watched vs unwatched cache writes.** The executor writes every node output
  via `cache._write(name, out)`. Need to confirm these interoperate correctly
  with the metrics/penalty/logging machinery (which uses "watched" attributes).
  Resolved/refined by #1.
- **Multi-output nodes.** `.split()` (tuple node → getitem refs) vs the
  `apply`-emitter pattern. Conventions for when to use which — mostly settled
  (apply when the second use is *behavior*; split for opaque tuple producers).
