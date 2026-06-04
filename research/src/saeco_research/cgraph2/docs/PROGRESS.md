# cgraph2 — progress log & resume protocol

## Resume protocol (for autonomous wake-ups)

I'm running autonomously overnight (Glen asleep ~till 8am 2026-06-04). On each
wake-up:

1. Read this file's **Current state** + **Plan** below.
2. `cd /home/g/wt/sae_components/active/branch-propagate`.
3. Continue the plan. Make the best call on any design question, record significant
   ones in `DECISIONS.md`, keep going — do not stop to ask.
4. Validate every change: cgraph2 demos + gated_graph smoke + ruff; GPU smoke
   through the real Trainer for arch changes.
5. Commit after each of #1/#2/#3 (and other meaningful milestones).
6. Re-arm the self-ping (ScheduleWakeup) so the loop survives, then continue.
7. Append to the **Log** below as you go.

Validation commands:
```
PYTHONPATH=research/src python -m saeco_research.cgraph2.demo   # + gated, funcstd_demo, inherit_subslot_demo, bubble_demo
PYTHONPATH=research/src python /tmp/smoke_gated_graph4.py        # CPU smoke (model+norm+losses+train)
ruff check research/src/saeco_research/cgraph2/ research/src/saeco_research/architectures/gated_graph/
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled PYTHONPATH=research/src:research/experiments timeout 280 python /tmp/observe.py  # GPU, real Trainer
```

## Plan

- **#1 — GraphSpec is the runnable context** (D-01). Make `GraphSpec` a `cl.Module`
  that registers its producers + has `forward_values`/`collect`/`forward`; retire
  `ValueGraph` into it. Keep bound sub-graphs as navigable children (D-02). Update
  `sae.py`/trainable to access handles via the graph. Test + GPU. **Commit.**
- **#2 — Trainable builds one TrainGraph.** The whole training computation is one
  graph exposing `total_loss`/`acts`/`L0`/per-term handles; `loss()` runs
  `g.forward_values` and saves through `g`. Total loss fused; weights applied in
  Python (coeffs are dynamic). **Commit.**
- **#3 — ActMetrics as GraphSpec + bubble-up metrics** (D-03). `ActMetrics` graph;
  metrics bubble up via `collect`; freq tracker bubble-up + deferred update.
  **Commit.**
- Then: polish, codex review passes, GNConfig normalizer (TODO), and whatever
  makes it most robust/elegant. Stretch: multi-input/DictBatch for transcoder
  `y_norm != x_norm`.

## Current state (start of overnight run, 2026-06-04)

- HEAD `e346973`: bubble-up sites + deferred forward_values committed.
- All green: cgraph2 demos (demo, gated, funcstd_demo, inherit_subslot_demo,
  bubble_demo), gated_graph CPU smoke, ruff. GPU run trains correctly (L2
  0.6->0.02, cosim 0.76->0.99, L0-targeting working).
- Architecture: model (`GatedModelSpec`) + `NormalizedModel` + losses are
  `GraphSpec`s; `GraphTrainable` hand-assembles a `ValueGraph` from loss/metric
  refs (the thing #1 replaces).

## Log

- (start) Scaffolded docs (INDEX/DECISIONS/PROGRESS), recorded D-01..D-03,
  committed prior bubble-up work. Beginning #1.
- **#1 DONE.** `GraphSpec` is now the runnable context: `forward_values`,
  `collect`, `forward`, `register_into`, `.of(**handles)` (dynamic aggregate),
  `apply_to`/`__call__`=apply. `ValueGraph` retired. Trainable builds its graph
  via `GraphSpec.of(...)` + `register_into(self)`, runs `self.graph.forward_values`.
  D-01 mechanism corrected (plain object + register_into, not nn.Module, because
  nn.Module.__call__ collides with g(x)=apply). All demos + CPU smoke + GPU green
  (trains identically). Next: #2.
- **#2 DONE.** Total loss is a fused graph handle (`total_loss`): each term *
  a scalar weight source-node (buffer synced from coeffs), summed (D-04). `loss()`
  runs one `forward_values`, `save(graph.total_loss)` is the backprop loss, terms
  saved unweighted for logging. L0-targeting's dynamic sparsity weight flows in;
  GPU run trains identically (L0 bends 3288->2774). Next: #3.
- **#3 DONE.** `ActMetrics` is a `GraphSpec` (acts -> L0/L1); a thin `_MetricsSite`
  passthrough (`metered(acts)`) is spliced on the gated model's acts and bubbled
  up via `collect` in the trainable, applied to the rebound (normalized) acts
  (D-05). L0/L1 are no longer hand-coded in the trainable -- they come from the
  model's site. Freq-tracker sites are collected + their EMA update deferred onto
  the loss pass (ready for resampling; no tracker spliced in the gated arch yet).
  All demos + CPU smoke + GPU green (L0-targeting bends L0 3288->2661). Next:
  polish + codex review.
- **Polish underway.** Committed ARCHITECTURE.md + research/tests/test_cgraph2.py
  (8 tests green). Resolved glen-inbox item 2 (class-vs-namespace -> GraphSpec is
  the runnable namespace). Ran codex review: `--base master` was too broad (whole
  branch diff incl. unrelated pre-existing files like matryoshka_clt.py); switched
  to a focused `codex exec` review of the 4 core files -> /tmp/codex_exec_review.txt.
  ON NEXT WAKE: read that file, address any real (P1/P2) findings, then continue
  polish (ideas: GraphSpec repr/structure viz; aux-model metrics bubble-up;
  GNConfig normalizer per TODO; below_* freq metrics with a spliced FreqTracker).
- **Self-review caught a real device bug (a8c1c05).** D-04's weight *buffers*
  go stale across `.to(device)` (buffers get reassigned, orphaning the graph
  node's capture on CPU; verified on GPU). Reverted to device-safe Python
  weighting; loss terms still share one fused pass. Superseded D-04 with the
  lesson: a graph node captures a *tensor*, and only `nn.Parameter` survives
  `.to()` in place -- never reference a bare registered buffer from a node.
  (The FreqTracker is fine: its `ref(self.freqs,())` is re-created each step
  post-`.to()` and `copy_` is in-place.) Codex review still running in bg.
- **Review pass done (D-06).** codex review (focused exec on the 4 core files)
  + self-review found 6 issues; all fixed: arg-slot identity (P1, the real one --
  binding a default-Input slot rewrote all Input), deferred-callback ordering +
  finally-reset (P1), weight buffers (P1, already done), sub_slot inheritance
  re-pointing (P2), register_into bare buffers (P2), empty-loss guard (P2). Added
  an arg-independence regression test (9 tests now). All green incl GPU.

## Tonight's deliverables (summary @ d6bb664)

#1 GraphSpec-is-runnable, #2 fused total (reverted to device-safe Python weighting
per D-04), #3 ActMetrics-as-graph + bubble-up metrics, the device-bug fix, 6
review-pass fixes (D-06, incl. the real arg-slot-identity P1), GraphSpec
repr/describe, and a live FreqTracker + below_* metrics in the gated arch. Docs:
INDEX/DECISIONS(D-01..D-06)/ARCHITECTURE/PROGRESS; 10 pytest in
research/tests/test_cgraph2.py. All green: pytest + demos + ruff + GPU (trains
identically, L0-targeting + freq EMA working).

## NEXT TASK (concrete) — GNConfig normalizer (Glen's TODO)

Make `AffineNormalizer` consume a `GNConfig` and match `GeneralizedNormalizer` for
the static-invertible configs. Keep the clean combined representation: `mu` (d,)
and `scale` (d,), with `forward=(x-mu)/scale`, `invert=x*scale+mu` (device-safe --
the module reads its own buffers). `prime(buffer, cfg)` computes, sequentially
like GeneralizedNormalizer: mu_e=x.mean(0) if PRIMED else 0; x-=mu_e;
mu_s=x.mean() if PRIMED else 0; x-=mu_s; std_e=x.std(0,unbiased=False)+eps if
PRIMED else 1; x/=std_e; std_s=(x.std(-1,unbiased=False)+eps).mean() if PRIMED
else 1; then `mu=mu_e+mu_s`, `scale=std_e*std_s`. Support {DONTUSE, PRIMED} for
all four aggs; raise NotImplementedError (pointing here) for RUNNING_AVG/LEARNED/
SAMPLE/BATCH_AVG (those need separate components / per-batch cache -- not the
static form). Signature `AffineNormalizer(d_data, cfg=GNConfig(), eps=...)`; the
arch passes `self.run_cfg.normalizer_cfg`. Add a test asserting it matches
`StaticInvertibleGeneralizedNormalizer` to ~1e-5 on the default cfg AND on
mu_e=PRIMED. Then update TODO.md. Validate (tests+demos+ruff+GPU), commit.
After that: aux-model metrics (L0_aux), or run the saeco library test suite to
confirm no repo-wide breakage (only research/ was touched).

## Log (cont.)

- **GNConfig normalizer DONE (D-07).** `AffineNormalizer` now takes
  `cfg=GNConfig()` and is the *static collapse* of
  `StaticInvertibleGeneralizedNormalizer`: priming replays the sequential
  `mu_e -> mu_s -> std_e -> std_s` estimation (each on running-normalized data,
  matching `GeneralizedNormalizer.prime_normalizer` incl. the `abs(std+eps)+1e-7`
  std_e form) and folds them into two buffers `mu = mu_e+mu_s`,
  `scale = std_e*std_s`; `forward=(x-mu)/scale`, `invert=x*scale+mu` (reads own
  buffers → device-safe). Static `{DONTUSE, PRIMED}` for all four terms;
  `NotImplementedError` (pointing at the class) for dynamic aggs
  (RUNNING_AVG/BATCH_AVG/LEARNED/SAMPLE) + sandwich. Arch passes
  `self.run_cfg.normalizer_cfg`. Two tests: matches the stock normalizer to <1e-5
  on default cfg **and** `mu_e=PRIMED` (forward + invert round-trip), and the
  dynamic-rejection guard. **11 pytest + 5 demos + ruff + GPU (real Trainer,
  gated_graph_gemma4: L2 0.57→0.02, cosim 0.76→0.99, L0-targeting active) all
  green.** TODO item checked off; D-07 recorded. Next candidates: aux-model
  metrics (L0_aux) bubble-up, or run the saeco library suite (`pytest tests/`) to
  confirm no repo-wide breakage (only research/ touched so far).
- **Library suite confirmed clean.** `pytest tests/` = **218 passed** — the
  research-side changes don't break the saeco library.
- **`GraphSpec.render()`** — the flat deduped DAG view: every node in the cone of
  all handles printed once, tagged with the handle name(s) pointing at it. On the
  gated trainable: the whole main+aux+3-losses+2-metrics computation is **23 nodes**
  with the encoder (`LinearMixed`) and the `AffineNormalizer` each appearing
  *once* (normalize shared across the model path and every loss). The fusion, made
  visible; complements `describe()`'s nested view. Test asserts a shared node
  renders once.
- **`field()` — the multi-input primitive (D-08).** A stateless `_Field` producer
  `ref(_Field(name), (Input,))`, interned per name, reading a named field off a
  *structured* input seeded at `Input` (dict→subscript / else getattr; fail-loud).
  Purely additive — the bare-tensor `Input` path is untouched, so all existing
  graphs/demos/the trainable are unchanged. Ships `multi_input_demo.py` (a
  transcoder-shaped weighted loss reading `input`/`target`/`weight` as three fused
  fields) in CI. This is the Option-B (DictBatch) surface from
  `docs/DESIGN-multi-input.md`; the *migration* (NormalizedModel `target`, the
  `total_loss` handle) is deliberately deferred for greenlight — it changes working
  behavior and has open questions.

## Overnight session 2 summary (@ HEAD after `field()`)

Picked up from the GNConfig-normalizer plan and delivered, all committed + green
(13 pytest, 6 demos, ruff, 218 library tests; GPU re-confirmed):

1. **GNConfig normalizer** (`a22e2ab`, D-07) — `AffineNormalizer(d, cfg=GNConfig())`
   is the static collapse of `StaticInvertibleGeneralizedNormalizer`; **codex
   brute-forced all 16 `{DONTUSE,PRIMED}` configs → value-identical**. Arch passes
   `run_cfg.normalizer_cfg`. Dynamic aggs raise `NotImplementedError`.
2. **Normalizer boundary doc + multi-input design note** (`1353516`) — codex review
   pass; `DESIGN-multi-input.md` proposal.
3. **`GraphSpec.render()`** (`d5afc92`) — deduped DAG visualization.
4. **`field()` multi-input primitive** (`d31ed10`, D-08) — additive; proves the
   `total_loss`/transcoder foundation without migrating the trainable.

**For Glen in the morning:** the one thing awaiting your call is the multi-input
*migration* (`docs/DESIGN-multi-input.md`, open questions at the bottom: DictBatch
type, per-loss vs vector weights, missing-`target` policy). The `field()`
primitive under it is built + demoed, so it's a quick "yes, do the migration like
that" / "adjust X" decision. Everything else (the normalizer TODO) is done.

## NEXT TASK (concrete) — multi-input migration, IF greenlit
Per `docs/DESIGN-multi-input.md`: land the `total_loss` handle first (smallest,
highest-value slice — pack `coeffs` into the batch as `w_*` fields, `total =
Σ field("w_k")*term_k`, resurrecting D-04 device-clean), then the transcoder
`target` field on `NormalizedModel`. Each its own commit, each validated GPU-
through-Trainer. If NOT greenlit: aux-model metrics (L0_aux) bubble-up, or retire
cgraph v1.
