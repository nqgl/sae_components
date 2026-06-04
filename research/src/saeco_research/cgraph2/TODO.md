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

- [~] **Multi-input.** Typed `DictBatch` contracts + `field(Batch.x)` + override
  DONE (D-09). **Trainable migration DONE (D-10):** `total_loss` is a graph handle
  (weights as `field("w_<k>")`, device-clean — the D-04 resurrection); transcoder
  `target` via `field("target")`; GPU trains identically. Remaining:
  - [ ] **(4) bubble-up logging recipe** — the output-side dual. The cache-write in
    `loss()` is still hand-listed; make it a bubble-up-built output recipe so
    inside-metrics appear without the outside naming them (D-03; Glen's note).
    Per refinement 2 + point-3, this is `forward_values` + projecting collected
    site `(name, handle)` pairs into a (boundary) DictBatch — no `produce` method.
  - [ ] **converter-as-GraphSpec** (`with_inputs`) — only if/when internal
    multi-output sharing needs fusion across a converter (refinement 2: internal
    composition is already just `arg`/`bind`/`sub_slot` ref-piping). Low priority.

- [ ] **Bubble-up metrics / freq-tracking / logging.** A site you pass `acts`
  into (L0, L1, acts, freqs, ...) that the trainable collects by walking the
  graph (apply-style) into a `{site-path: site}` dict, each site exposing handles
  to `save()` in the `forward_values` block. Freq tracker: `freqs` (input) +
  `updated_freqs = f(freqs, acts)`; the trainable bubbles trackers up and defers
  `tracker.update(save(tracker.updated_freqs))` to the end of the `with` block
  (ideally inline via deferred execution on the context manager). Same shape for
  bubble-up logging (`{site-path: log-handle}`).
