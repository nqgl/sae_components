# cgraph — completed

Move items here from [TASKS.md](./TASKS.md) as they land. Newest first.

## 2026-06-02

- **v2 gated arch port** (`cgraph2/gated.py`). The gated SAE (main + aux + 3
  losses + `GraphTrainable`) on the relational core — and it's simpler than the
  v1 port: no builder, no `apply(g, x)`, no `SAEApplied` bundle threading `g`.
  The models build interned nodes that read the global `Input`; `enc_gate()` is
  one node by construction, so referencing it from the main model, aux model, and
  sparsity loss shares it automatically. Produces the *identical* fused graph as
  v1 (encoder once feeding gate+mag; the gate activation feeds Thresh→main,
  Detach→aux, L1→sparsity), encoder runs once, trains 2.69 → 0.34. Also: re-applied
  `arg(default=Input | param | handle)` and added the v2 `Nest` + param-default to
  `cgraph2/demo.py` (value slots defaulting to shared handles -> encoder once).

- **cgraph v2 — relational / interned graphs (#7), first increment**
  (`cgraph2/` — `core.py`, `spec.py`, `demo.py`). Clean rewrite in its own
  directory. A node is an immutable `ValueRef(producer, inputs)` **interned by
  structure** in a global table, so identical structure is the *same object* —
  fusion is global and automatic with **no mutable builder and no inline/re-emit**.
  One global `Input` node; `x` defaults to it (no `g.input()`). `GraphSpec` is the
  only authoring surface; binding producer slots is structural re-interning (which
  re-interns globally, so bound graphs sharing structure share nodes). Demo proves:
  `produce(acts)` runs only the encoder cone; recon-loss + penalty(acts) runs the
  encoder **once** just by referencing handles (the v1 `Nest` case, now trivial);
  **two separately-constructed specs with the same producers share the same
  interned `acts` node**; different producers → different nodes; `op(torch.relu)`
  fuses by function id. v1 stays green alongside. Open (per TASKS #7): intern-table
  lifetime/scoping, multi-input (`DictBatch`), porting the gated arch + losses +
  trainable onto v2.

- **Consing keyed on function identity, not labels** (`slots.py`). Fixed a
  stringly-typed footgun: arithmetic ops were consed off their display label
  (`"Mul"`), so two unrelated ops sharing a label + inputs would have wrongly
  merged. Now arithmetic ops are module-level functions and the cons key is
  `id(fn)` (scalar folds key on `(id(fn), const)`). Verified: `a*b` still fuses
  (shared `_mul`); same-label different-function do NOT collide.

- **Graph producers auto-inline on application** (`slots.py`). `_apply` now
  inlines a cgraph `Graph` (recursively, hash-consing) instead of adding an opaque
  node — so two *distinct* frozen subgraphs that share an encoder fuse (Glen's
  `Nest` test: `produce(_sae.output)` + `produce(_sae.acts)` → encoder once). This
  makes frozen-graph fusion the default (was opt-in `inline=True`). Updated
  `fusion_stress` (case 3 now ONCE) and `slots_demo` (count encoder nodes, since
  the model Graph flattens). Tradeoff: subgraphs no longer print as nested blocks
  (they're flat/fused) — fine given the move to handles + the relational direction.

- **Output-handles on `GraphSpec` — the cache replacement** (`namespace.py`,
  `slots.py`, `handles_demo.py`, `CACHE_TO_HANDLES.md`). A `GraphSpec` instance
  now exposes **every named body value as a grabbable handle** (`sae.acts`,
  `sae.recon`) via `__getattr__` over the inlined node-map (`inline` returns the
  full template-node → target-ref map; class-body refs are stripped from the
  class dict so instances serve inlined handles). `ValueRef.build()` /
  `spec.produce(ref)` freeze a Graph targeting one handle — it runs only that
  handle's **cone** (`produce(sae.acts)` runs the encoder, skips the decoder).
  Splicing a handle into another graph (the penalty: `L1(sae.acts)`) fuses with
  the model (encoder once). This is the typed, first-class replacement for the
  cache's "reach a value out of the middle of the forward pass". The cache→handle
  mapping (what's replaced, what's open: logging, resampler, metrics, multi-output
  `DictBatch`) is assessed in `CACHE_TO_HANDLES.md`.

- **#3 (first increment) — gated arch on the slot model** (`gated_graph.py`).
  Parallels saeco: `GraphLoss`/`GraphL2Loss`/`GraphSparsityLoss` ~ `Loss`/`L2Loss`/
  `SparsityPenaltyLoss` (each emits its loss *term* as a node); `GraphTrainable
  (Trainable)` ~ `Trainable` with `loss()` building ONE fused graph; `GatedGraph`
  ~ the `Gated` arch (`enc_mag`/`enc_gate`/`gated_model`/`model_aux`, encoder
  shared by mag+gate, enc_gate shared by main+aux). **Full fusion proven**: the
  printout shows `n4 = relu` (the gate activation) feeding `Thresh` (main),
  `Detach`→decoder (aux), and `L1` (sparsity) — ONE node across main + aux + all
  3 losses; encoder runs once; trains 2.69 → 0.34. Replaces every `ReuseForward`
  in the original gated arch.
  - **Surfaced + fixed the dedup issue in the wild**: `enc_gate`'s `nn.ReLU()`
    instances didn't fuse (3 relus); switching to the function ref `torch.relu`
    fused them to one — concrete validation of the PROD_READINESS #16 guidance.
  - Not yet: real `Initializer` modules, real `Trainable` contract
    (metrics/resampler/coeffs/normalization), descriptor integration
    (`@model_prop` returning graphs). Leaf modules are stand-in Linears.

- **Dedup-vs-override tension test** (`dedup_test.py`) + small niceties (`Arg`/`Op`
  annotation aliases; raw bindings auto-wrap, so `nl2=nn.GELU()` works). Showed:
  two distinct `g(nn.ReLU())` slots are independently overridable
  (`G(x) != G(x, nl2=GELU) != G(x, nl1=GELU, nl2=GELU)`) but their identical
  defaults recompute (ReLU twice); a shared instance dedups (ReLU once) but
  can't be overridden independently. Logged as PROD_READINESS #16.

- **Unified `g(...)` restored in the slot layer** (`slots.py`). Replaces the
  `g.op(...)`/`g.param(...)` split with one overloaded entry point that dispatches
  on what you hand it (matching core's `g(producer, *inputs)`):
  - `g(value)` (Tensor / Parameter)       -> `ValueRef`
  - `g(producer)` (module / callable)     -> `OpRef` (apply later)
  - `g(producer, *inputs)`                -> `ValueRef` (apply now)
  Type-visible (pyright reveals the right type per form). `op`/`param`/`input`
  stay as internal helpers. So `g(enc, x)`, `g(sae)`, `g(some_param)` all read
  uniformly.

- **Concrete slot defaults via `g(...)` in the namespace** (`namespace.py`,
  `slots.py`). CORRECTION of a wrong earlier claim: `compare: OpRef = g(nn.ReLU())`
  *does* work as an overridable, type-visible default — no `__missing__` magic.
  The fix: detect slots by **annotation** (any annotated `ValueRef`/`OpRef`
  field is a slot), and use the slot's **producer identity** (a marker OR the
  concrete default producer) as the substitution key — the metaclass links
  name->ref via `vars(cls)`. `inline` substitutes by `producers.get(producer,
  producer)`, so an unbound concrete default passes through and an override
  replaces it. `g` is injected into the body (with a module-level placeholder so
  it type-checks). Validated: omitting `compare` uses the `g(...)` default;
  `compare=mae` changes the graph; pyright marks `compare` optional and rejects a
  bad binding. (Only the *annotation-only* form `x: ValueRef` with no `= ...`
  still needs `__missing__`.)

- **Inline cgraph Graphs to fuse through frozen boundaries** (`slots.py`,
  `fusion_stress.py` case 4). `g.op(graph, inline=True)` flattens a built Graph's
  nodes into the builder on application (recursively), so two *distinct*
  frozen subgraphs with identical internals fuse (the case-3 failure). Opaque
  (default) keeps the subgraph as a black box — a cache-scope boundary that
  prints as a nested block. So it's a per-`op` choice: fuse-through (`inline`) vs
  boundary (default). (Open: whether to auto-inline cgraph Graphs by default —
  it'd fuse everywhere but dissolve the nested-print/cache boundary.)

- **Slot defaults** (`namespace.py`, `slots.py`). `op_slot(default=<producer>)` /
  `arg(default=<value>)` make a slot optional with a fallback used when unbound at
  construction (e.g. `compare: OpRef = op_slot(default=nn.ReLU())`). Type-visible:
  the field specifier's `default` makes the construction param optional in pyright.
  (The bare `compare: OpRef = nn.ReLU()` form still needs the deferred
  `__missing__` namespace — it would otherwise be a raw module, not a slot.)

- **#6 slot model — `dataclass_transform` namespace** (`namespace.py`,
  `namespace_demo.py`). A graph template written as a class:
  ```python
  class Loss(GraphSpec):
      x: ValueRef = arg()        # value-input slot
      model: OpRef = op_slot()   # producer slot
      compare: OpRef = op_slot()
      y_pred = model(x)
      output = compare(x, y_pred)
  ```
  Metaclass `__prepare__` opens a recording builder for the body; constructing
  `Loss(x=…, model=…, compare=…)` binds the slots and **inlines the template into
  the builder of the bound refs** (so nested templates fuse — model bound into two
  losses runs once, validated). Type-visible via `dataclass_transform`: pyright
  shows the construction signature, infers the result `Loss`, and **rejects**
  `model="bad"` / missing slots. `arg()`/`op_slot()` are the field specifiers
  (the body still uses the same slot handles as the explicit form). Slot
  inlining (`GraphDef.inline`) added to `slots.py`. Deferred: the annotation-only
  form (`x: ValueRef` with no `= arg()`, needs a `__missing__` namespace),
  multi value-input, and `Op[Args,T]` arity-typed producer slots.

- **Thread `T` through the slot types** (`slots.py`). `ValueRef[T]` / `OpRef[T]`;
  `g.op(fn) -> OpRef[T]`, applying -> `ValueRef[T]`, `g.input() -> ValueRef[Tensor]`.
  pyright now reveals `ValueRef[Tensor]` instead of `ValueRef[Unknown]` for op
  results (module producers carry saeco's `Tensor | list[Tensor]` union).

- **Recursive subgraph printing** (`core.py`). `Graph.render()` (used by `__repr__`)
  recurses into subgraph (Graph-producer) nodes, printing each indented; pass
  `recurse=False` for the flat one-level view.

- **Deep-nesting fusion stress test** (`fusion_stress.py`). Answers "does
  `A(B(C(x->d)) + B(F(x->d)))` run `x->d` once?": YES when composition shares a
  builder (case 1) or reuses the same frozen subgraph object (case 2); only
  DISTINCT frozen subgraphs with identical internals run it twice (case 3,
  opaque boundary). So fusion reaches through arbitrary depth as long as
  composition emits into a shared builder — which the slot model / namespace does
  by construction, so no inline+CSE pass is needed.

- **#6 slot model — explicit API** (`slots.py`, `slots_demo.py`). `ValueRef[T]`
  (value handle, was `Ref`) + `OpRef` (applyable producer handle, was `Op`):
  `g.input()`/`g.param()` make value slots, `g.op(producer)` makes a producer
  slot, applying an OpRef adds a node. **A graph template is a function
  `(g, **producer_slots) -> ValueRef`** — producer slots are typed params, so
  binding is `functools.partial`. Proven:
  - *Runtime*: a `Loss` with a `model` producer slot; binding the SAME model into
    two losses and fusing runs the model **once** (one shared `Graph` node), and
    it trains.
  - *Type-visibility (pyright)*: `build_loss` reveals as
    `(g, *, model: OpRef, compare: OpRef) -> ValueRef`, `model(x)` infers
    `ValueRef`, and `model="..."` is rejected. The `model(x)` that was red under
    the namespace-returns-Graph form is now green — no metaclass needed.

  Open refinement: output type is `ValueRef[Unknown]` (thread `T` through
  `OpRef`/`op`). Next: the `dataclass_transform` class-namespace sugar over this.

- **attrs + factory-classmethod refactor.** The plain classes are now `@define`:
  `_Node`/`Ref`/`GraphDef`/`Op` (core), `AffineApplied`/`AffineNormalizer` (demo),
  `ScratchArch` (spike). Init logic moved to factory classmethods (`Op.create`,
  `ScratchArch.create`); identity-sensitive ones use `eq=False`. nn.Module
  subclasses (`Graph`, leaf modules, `MockTrainable`) stay as torch modules.
  Graph's structural build moved into `GraphDef.build` (toposort + naming) so
  `Graph.__init__` just stores + registers. All demos green.

- **`g.output =` + namespace finalization** (answers "why `g.build(node)`?").
  `@g.namespace` now learns names, reads the `g.output` designated in the body,
  builds, and *returns the Graph* — so the decorated class **is** the graph.
  `g.build(ref)` stays as the non-namespace path. **Finding:** a class decorator
  that returns a `Graph` is opaque to pyright (it keeps the name typed as the
  class, so `model(x)` looks wrong) — runtime-clean but NOT type-visible. This is
  the concrete motivation for the slot-model + `dataclass_transform` design
  (which type checkers *do* honor). Logged for the slot-model task.

- **Automatic fusion via hash-consing** (`core.py`, `fusion_spike.py`). The
  builder now CSEs at construction: `g(producer, *inputs)` returns an existing
  node when `(producer, input-nodes)` matches (key = op-kind for `Op`s, instance
  id for modules/params). Two model builders authored with **no awareness of
  each other** (`emit_main`/`emit_aux`) — sharing only the builder, the input
  ref, and the `enc` instance — fuse automatically: the whole shared `enc→relu`
  prefix collapses to single nodes, encoder runs once. `fresh=True` opts an
  impure (stochastic/stateful) producer out. All prior demos unaffected (consing
  only merges genuinely-identical nodes). Resolves the "assemble subgraphs
  without awareness, redundancies fuse away" goal — *within a shared builder*.

## 2026-06-01

- **#4 design spike — total loss as one fused graph** (`megaloss_spike.py`).
  Validated the megagraph plan: with main + aux + penalty + per-loss math +
  weighted sum in one DAG sharing an encoder node, the encoder runs **once**
  (vs **twice** under naive opaque sub-graph nesting — proving dedup is by node
  identity, so fusion must share refs or CSE). Penalty expressed as a value node
  on an edge (retires `cache.search`); aux's detach is an explicit `.detach()`
  edge (grad doesn't leak to the shared encoder). Trains: 1.38 → 0.066.
  Implementation into the real `Loss`/`Trainable` is still pending (TASKS #4).

- **#1 — mock-Trainable wiring** (`trainable_spike.py`). End-to-end:
  - `ScratchArch.make_trainable()` returns `MockTrainable(Trainable)` whose
    `self.model` is a normalizing graph wrapping a nested unnormalized SAE graph
    (the agreed structure) — not `NormalizedIO`. The override seam is exactly
    what a real `Architecture` subclass would do.
  - A real `co.L1Penalty` node buried in the inner graph writes
    `sparsity_penalty`; real `co.SparsityPenaltyLoss.search(...)` finds **exactly
    one** — the executor's per-node `cache[name]` discipline preserves the nested
    subcache tree (resolves PROD_READINESS #4).
  - Surfaced + demonstrated a non-obvious dependency: today's loss layer re-runs
    the model per loss and leans on `ReuseForward` to dedup the penalty write
    across calls. Naive per-loss re-run double-counts; the graph-native fix is
    **run once, losses read the shared cache** (PROD_READINESS #10).
  - Trains: loss `0.90 → 0.056`; the sparsity penalty alone produces encoder
    gradients (genuinely wired through the cache).

- **Nit: `Op` label optional** — `Op(torch.exp)` auto-labels "exp" via the
  callable's name; module producer labels use `nn.Module._get_name()`.

- **Core engine** (`core.py`, ~150 lines).
  - `Ref[T]` — typed node handle; arithmetic operators (`*`/`+`/`-`/`/`) build
    fan-in op nodes; `.split(n)` and `.detach()` per-edge transforms.
  - `GraphDef` — mutable builder; `g.input`, `g(producer, *inputs)`,
    `@g.namespace` for naming, `g.build(output)`.
  - `Graph` — frozen runnable `cl.Module`; topological evaluation with
    **results keyed by node identity (dedup) + cycle detection**; per-node
    subcache via `cache[name]` + output written via `cache._write`; registers
    parametric children for `state_dict`; adjacency `__repr__`.
  - Names resolved into a local map at build (nodes stay immutable / reusable
    across builds — fixed a latent build-time mutation bug).

- **Gated encoder rewrite validated** (`demo.py::gated_demo`). The shared
  encoder is one node with two out-edges (gate + magnitude). Matches a
  plain-torch reference; **encoder ran exactly once** (vs. 3 `ReuseForward`
  wrappers + a two-`Seq` split in today's `ClassicGated`).

- **Affine normalizer rewrite validated** (`demo.py::normalizer_demo`). The
  `apply(g, x) -> AffineApplied` bundle owns `invert`; scale & shift each feed
  normalize *and* invert but are **computed once each**. No `cache.scale = ...`
  write-back, no `ReuseForward`. A nested SAE graph composes as a single node.
