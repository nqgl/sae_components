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
