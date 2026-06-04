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

- [ ] **Bubble-up metrics / freq-tracking / logging.** A site you pass `acts`
  into (L0, L1, acts, freqs, ...) that the trainable collects by walking the
  graph (apply-style) into a `{site-path: site}` dict, each site exposing handles
  to `save()` in the `forward_values` block. Freq tracker: `freqs` (input) +
  `updated_freqs = f(freqs, acts)`; the trainable bubbles trackers up and defers
  `tracker.update(save(tracker.updated_freqs))` to the end of the `with` block
  (ideally inline via deferred execution on the context manager). Same shape for
  bubble-up logging (`{site-path: log-handle}`).
