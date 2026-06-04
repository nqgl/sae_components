# cgraph2 — todo

- [ ] **Normalizer: match `GeneralizedNormalizer` + consume `GNConfig`.**
  `AffineNormalizer` (in `sae.py`) currently reimplements only the *default*
  static-invertible affine map (center by scalar mean, scale by per-feature std ×
  scalar std), matching the old normalizer to ~7e-6. Generalize it to cover the
  full `GNConfig` behavior (the `mu_e`/`mu_s`/`std_e`/`std_s` aggregations —
  DONTUSE / PRIMED / RUNNING_AVG / BATCH_AVG / LEARNED / SAMPLE, sandwich) and
  take a `GNConfig` so it's a drop-in for the configurations people actually
  sweep. Keep the clean implementation (no cache-routed `mu_e(x)` methods).

- [ ] **Bubble-up metrics / freq-tracking / logging.** A site you pass `acts`
  into (L0, L1, acts, freqs, ...) that the trainable collects by walking the
  graph (apply-style) into a `{site-path: site}` dict, each site exposing handles
  to `save()` in the `forward_values` block. Freq tracker: `freqs` (input) +
  `updated_freqs = f(freqs, acts)`; the trainable bubbles trackers up and defers
  `tracker.update(save(tracker.updated_freqs))` to the end of the `with` block
  (ideally inline via deferred execution on the context manager). Same shape for
  bubble-up logging (`{site-path: log-handle}`).
