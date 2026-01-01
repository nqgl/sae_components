# Evaluation API Changes (2025-12) — Breaking

This document describes **breaking** API changes made to `saeco.evaluation` to improve naming clarity and ergonomics.

## Summary of changes

### Renames (types / files)

| Old | New |
|---|---|
| `saeco.evaluation.storage.saved_acts_config.CachingConfig` | `saeco.evaluation.storage.cache_config.CacheConfig` |
| `saeco.evaluation.storage.saved_acts.SavedActs` | `saeco.evaluation.storage.cached_acts.CachedActs` |
| `saved_acts_config.py` | `cache_config.py` |
| `saved_acts.py` | `cached_acts.py` |

### Renames (Evaluation fields / properties)

| Old | New |
|---|---|
| `Evaluation.saved_acts` | `Evaluation.cached_acts` |
| `Evaluation.cuda` (device) | `Evaluation.device` |
| `Evaluation.docs` | `Evaluation.samples` |
| `Evaluation.num_docs` | `Evaluation.num_samples` |
| `Evaluation.docstrs` | `Evaluation.token_strs` (token strings) OR `Evaluation.text` (decoded text) |
| `Evaluation.cached_call` | `Evaluation.cached` |
| `Evaluation.metadatas` | `Evaluation.metadata_store` |
| `Evaluation.artifacts` | `Evaluation.artifact_store` |
| `Evaluation.filters` | `Evaluation.filter_store` |
| `Evaluation.open_filtered(name)` | `Evaluation.open_filter(name)` |
| `Evaluation.get_feature(x)` | `Evaluation.feature(x)` |
| `Evaluation.chill_top_activations_and_metadatas(...)` | `Evaluation.top_activations(...)` |

### New API additions

- `eval.text[idx]` returns decoded text (`str` or `list[str]`).
- `eval.token_strs[idx]` returns token strings (`list[str]` / `list[list[str]]`).
- `eval.metadata[key]` returns metadata aligned to the current evaluation doc space.

## How to update downstream code (search/replace)

### CacheConfig / CachedActs
- Replace imports:
  - `from saeco.evaluation.storage.saved_acts_config import CachingConfig`
    -> `from saeco.evaluation.storage.cache_config import CacheConfig`
  - `from saeco.evaluation.storage.saved_acts import SavedActs`
    -> `from saeco.evaluation.storage.cached_acts import CachedActs`

### Opening evaluations
- `Evaluation.from_cache_name(x)` -> `Evaluation.open_cache(x)`
- `Evaluation.from_model_path(p, ...)` -> `Evaluation.open_model(p, averaged_weights=...)`

### Device usage
- Replace `eval.cuda` (as a device) with `eval.device`.
  - Example: `t.to(eval.cuda)` -> `t.to(eval.device)`

### Samples vs text
- `eval.docs[...]` -> `eval.samples[...]`
- `eval.num_docs` -> `eval.num_samples`
- `eval.docstrs[...]`:
  - if you want token strings: `eval.token_strs[...]`
  - if you want decoded text: `eval.text[...]`

### Cached calls
- `eval.cached_call.some_method(...)` -> `eval.cached.some_method(...)`

### Filters / Metadata / Artifacts
- `eval.open_filtered(name)` -> `eval.open_filter(name)`
- `eval.filters[name]` -> `eval.filter_store[name]`
- `eval.metadatas[name]` -> `eval.metadata_store[name]`
- `eval.artifacts[name]` -> `eval.artifact_store[name]`

### Features / top activations
- `eval.get_feature(7)` -> `eval.feature(7)`
- `eval.chill_top_activations_and_metadatas(7, k=10)` -> `eval.top_activations(7, k=10)`

## Notes
- Compatibility shims were intentionally NOT provided to keep the API clean and force dependent code to update.
