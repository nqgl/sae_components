# cgraph2 — todo

- [x] **Normalizer: match `GeneralizedNormalizer` + consume `GNConfig`.** DONE.
  `AffineNormalizer(d_data, cfg=GNConfig(), eps=1e-7)` now consumes a `GNConfig`
  and folds the four sequential factors (`mu_e -> mu_s -> std_e -> std_s`) into
  the two primed buffers `mu`/`scale` — a clean static collapse of
  `StaticInvertibleGeneralizedNormalizer` (no cache-routed `mu_e(x)` methods).
  Covers the *static* aggregations `{DONTUSE, PRIMED}` for all four terms;
  `NotImplementedError` (pointing back here) for the dynamic ones (RUNNING_AVG /
  BATCH_AVG / LEARNED / SAMPLE) and `sandwich` — those aren't a single static
  affine map (D-07). The arch passes `self.run_cfg.normalizer_cfg`. Tests assert
  it matches the stock normalizer to <1e-5 on the default cfg *and* on
  `mu_e=PRIMED`, plus the dynamic-rejection guard. Validated: 11 pytest + 5 demos
  + ruff + GPU through the real Trainer (trains identically).
  *Remaining (separate, deferred):* the dynamic aggregations would need either
  per-batch cache routing (BATCH_AVG/SAMPLE), a stateful EMA component
  (RUNNING_AVG — same shape as the FreqTracker bubble-up), or a learned param
  (LEARNED). Not a static affine map, so out of scope for `AffineNormalizer`.

- [x] **D-04 device bug: full understanding (Glen, flagged important).** DONE — D-17.
  The complete model (empirically verified; CPU-reproducible via `.to(float64)`):
  `nn.Module._apply` swaps a `Parameter`'s `.data` in place (safe to capture) but
  *reassigns* registered buffers (a captured bare tensor goes stale; the old
  `register_into` behavior even manufactured a lying state_dict duplicate). Fix:
  slot-bound/default bare tensors are auto-wrapped in a `_TensorSource` (a module
  reading its *own* live buffer — device-safe, fuses by tensor identity); raw
  `ref(tensor, ())` now fails loud at registration with the safe forms named.
  Four-part regression test; GPU train-identical.

- [~] **Multi-input.** Typed `DictBatch` contracts + `field(Batch.x)` + override
  DONE (D-09). **Trainable migration DONE (D-10):** `total_loss` is a graph handle
  (weights now one seeded `weights: dict[str, float]` field indexed by loss name —
  D-16; device-clean, the D-04 resurrection); transcoder `target` via
  `field(SAEInputs.target)`; GPU trains identically. Remaining:
  - [ ] **(4) bubble-up logging recipe** — the output-side dual. The cache-write in
    `loss()` is still hand-listed; make it a bubble-up-built output recipe so
    inside-metrics appear without the outside naming them (D-03; Glen's note).
    Per refinement 2 + point-3, this is `forward_values` + projecting collected
    site `(name, handle)` pairs into a (boundary) DictBatch — no `produce` method.
  - [ ] **converter-as-GraphSpec** (`with_inputs`) — only if/when internal
    multi-output sharing needs fusion across a converter (refinement 2: internal
    composition is already just `arg`/`bind`/`sub_slot` ref-piping). Low priority.

- [~] **Bubble-up metrics / freq-tracking / logging.** Metrics + freq-tracking
  DONE (D-03/D-05/D-13): `_MetricsSite` bubble-up, `collect(FreqTracker)`, and the
  `@deferred_execution` training-gated EMA update (`tracker.update(tracker.updated(
  acts))` inline in the block). Remaining: the bubble-up *logging* half — same as
  the multi-input "(4) bubble-up logging recipe" item above (`{site-path:
  log-handle}` projected at the boundary, so `loss()` stops hand-listing the
  cache writes).
