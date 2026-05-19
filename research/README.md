# saeco-research

Personal research scratch space for the [`saeco`](../README.md) library.

This package contains:

- **`saeco_research/architectures/`** — in-flight architecture experiments. Some are stable, some are half-finished, some are abandoned. Treat as a working notebook.
- **`experiments/`** — runnable training/sweep scripts. Reference these for "how would I actually use saeco?" but don't expect them to all work out of the box.
- **`scripts/`** — exploration scripts and one-off analyses.

## Status: no API guarantees

Nothing in this package is API. Modules get renamed, deleted, refactored without notice. Imports may be broken at any commit. If you depend on something here and want it stable, ask me to promote it to the `saeco` library proper.

## Install

From the repo root, install the dependency chain (`sweepable` → `saeco`
→ `saeco-research`):

```bash
pip install -e ./sweepable
pip install -e .
pip install -e ./research
```

Then `from saeco_research.architectures.<name> import ...` works alongside `from saeco.architectures.vanilla import ...`.
