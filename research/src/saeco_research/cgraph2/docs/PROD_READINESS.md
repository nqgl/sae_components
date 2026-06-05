# cgraph2 — production-readiness carryover

Forward-looking correctness/completeness concerns that must be resolved (or
consciously accepted) before any cgraph2 module is promoted from `research/` into
`src/saeco/`. Distilled from the **v1** `cgraph/PROD_READINESS.md` and
`cgraph/CACHE_TO_HANDLES.md` when v1 was retired (2026-06-05) — only the rows that
are still *live* under the v2 relational/interned design are carried here; v1's
resolved-or-superseded rows (the full table) remain in git history at the v1
deletion commit. Add to this whenever a corner is cut.

## Carried-over open concerns

### 1. State-dict key stability & shared-module registration
*(v1 #1, #2, #3)*
- Interned `ValueRef` nodes not bound to a class attr get auto names; these become
  `state_dict` keys / cache paths. A rename/reorder must not silently break
  checkpoints. Needs a defined, stable key scheme.
- One `nn.Module` instance producing 2+ nodes (weight tying via shared producers)
  would `add_module` it under two names → duplicated/aliased `state_dict` entries.
  Detect shared instances and register once / alias, or define tying semantics
  explicitly.
- Node-output writes must be collision-safe against watched cache field names
  (`L1`, `L0`, `loss`, `cosim`, `L2_loss`, ...) — a node named like a watched
  field could hijack that slot. Verify the v2 handle/cache mechanism guards this.

### 2. Impure / stateful producers under interning
*(v1 #13, #15, #16; CACHE_TO_HANDLES #5)* — **the structural analog of the
`funcstd.standardize` soundness contract.** Interning/hash-consing assumes producer
purity:
- A stateful/stochastic producer (EMA/freq tracker, dropout, noise) called twice
  with the same input is merged to ONE execution — changes semantics. Need an
  `impure`/`fresh=` opt-out (or detect-and-error when an impure-marked producer
  would be consed).
- "Effect" nodes that *must* run but whose value no one consumes (e.g. an EMA
  tracker update) are not pulled in by a pure-value toposort. Need an explicit
  effect edge or to make them consumed value nodes.
- Dedup keys on producer *identity*: two distinct parameter-free instances
  (`nn.ReLU()` vs `nn.ReLU()`) don't dedup though semantically equal; sharing the
  instance dedups but couples the slots (can't override independently). Author
  guidance + optional canonicalization pass for parameter-free producers.
- Function-level fusion soundness lives in `funcstd.standardize` (its key must
  never collide two behaviorally-distinct functions). Keep that contract and this
  one in sync.

### 3. Resampler + save/load + sweep integration unverified
*(v1 #7; CACHE_TO_HANDLES #3)*
A `GraphSpec`/`GraphTrainable` is a `cl.Module` so it *should* introspect, but the
resampler's module-walking + activation/freq reads, checkpoint save/load round-trip,
and a real sweep have not been run end-to-end against a graph. Rephrase the
resampler to consume the `acts` handle (and `produce(acts)` for encode-only
activations cheaply). Run a real architecture through resample + save/load + a
sweep before promotion.

### 4. Trainable contract & loss space
*(v1 #11, #12)* — partially built now (`GraphTrainable`/`TrainGraph`), so this is
*verify*, not *design*:
- Confirm the graph trainable matches the real `Trainable` contract:
  `rearrange`/`dearrange`, auto metrics + `CosineSimilarityLoss`/`L2Loss`, coeffs
  from `train_cfg`, resampler hookup.
- Decide the loss space deliberately (data-space vs normalized-space objective)
  and confirm losses compare in the intended space. With the embedded normalizer
  (Step 2), the denormalized output must be numerically identical to
  `normalizer.invert`.

## Resolved in v2 (do not re-carry)
- Multi-output / `DictBatch` returns + a true `total_loss` handle — built (D-09,
  D-10).
- Predictable handle interfaces / handle contract — `SAEGraphSpec` declares the
  model contract (`acts`, `recon`, ...).
- Cross-call `ReuseForward` dedup / penalty-as-value-node — subsumed by interning
  (shared nodes run once by identity) + the fused `total_loss` DAG.
