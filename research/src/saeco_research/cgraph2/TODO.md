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

- [ ] **D-04 device bug: full understanding (Glen, flagged important).** Come back
  to this *thoroughly* once the multi-input work settles. The class of bug —
  `.to(device)` reassigns buffers and orphans a graph node's captured tensor on the
  old device, "working" only via silent cross-device scalar promotion — is exactly
  the kind that causes very bad subtle bugs later. We want a solid, complete model
  of: precisely which producers are safe to reference from a node (nn.Parameter:
  yes, in-place `.data` swap; bare buffer: NO, reassigned; a module reading its own
  live buffer: yes) and *why*; whether `register_into` / interning can detect or
  prevent capturing a bare buffer; and whether a guard/assert at the boundary
  (e.g. forbid `ref(buffer, ())` for a registered buffer) would catch it loudly.
  See D-04 in DECISIONS.md for the original catch.

- [ ] **Multi-input: trainable migration + converter-as-GraphSpec.** Typed
  `DictBatch` contracts + `field(Batch.x)` + override are DONE (D-09). Remaining:
  (2) `total_loss` handle, (3) transcoder `target`, (4) the multi-output/logging
  recipe — and Glen's **converter-as-GraphSpec** (a converter is a graph, composed
  by binding `field(name)` reads to the converter's named handles; verified to
  preserve fusion where a Python-function converter does not — see
  `docs/DESIGN-multi-input.md`). The converter-as-graph *is* the multi-output recipe
  (inputs/outputs unify), so (4) and the converter design are one piece.

- [ ] **Bubble-up metrics / freq-tracking / logging.** A site you pass `acts`
  into (L0, L1, acts, freqs, ...) that the trainable collects by walking the
  graph (apply-style) into a `{site-path: site}` dict, each site exposing handles
  to `save()` in the `forward_values` block. Freq tracker: `freqs` (input) +
  `updated_freqs = f(freqs, acts)`; the trainable bubbles trackers up and defers
  `tracker.update(save(tracker.updated_freqs))` to the end of the `with` block
  (ideally inline via deferred execution on the context manager). Same shape for
  bubble-up logging (`{site-path: log-handle}`).
