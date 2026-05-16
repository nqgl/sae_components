# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Monorepo of three packages

This repo ships three independently-installable packages with a strict one-way
dependency chain. **Install order matters** (none are on PyPI yet):

```bash
pip install -e ./sweepable    # the sweep-config DSL; depends only on pydantic + paramsight
pip install -e .              # `saeco`, the library; depends on sweepable
pip install -e ./research     # `saeco-research`; depends on saeco
pip install -e ".[dev]"       # pytest, ruff, pre-commit
```

- `sweepable/` — standalone, **zero saeco deps**, git-subtree-extractable. Source: `sweepable/src/sweepable/`.
- `src/saeco/` — the polished library with an API contract.
- `research/` — in-flight scratch (experimental architectures, evaluation API, analysis GUIs). **No API stability, not packaged, not CI-linted.** Importable as `saeco_research.*`.

Requires **Python 3.13+** (PEP 695 generics are used pervasively; do not target older).

## Commands

```bash
# Tests — three separate suites (CI runs all three):
pytest tests/                              # saeco library
pytest sweepable/tests/                    # sweepable
pytest research/tests/                     # tests for moved-out modules
pytest tests/sweeps/test_sweepable_config.py::test_name -q   # single test

# Lint (CI gate — pyright is NOT a gate and is not expected to pass):
ruff check src tests sweepable/src sweepable/tests
ruff format src tests sweepable          # apply formatting
ruff check <paths> --fix                 # safe autofixes (B-rules unfixable by config)
```

`ruff` selects `E,F,I,N,UP,B,A,C4,PT`. `sweepable/` is lint-clean and is the
meaningful gate. `src/`+`tests/` carries legacy lint (naming/E501/cell-style
files) being reduced incrementally — a non-green count there is expected and
not a regression you introduced; keep new and changed code clean.

## Architecture system (the core abstraction)

An SAE is defined as an `Architecture[ConfigT]` subclass (see
`src/saeco/architectures/vanilla/vanilla_model.py` and `gated/`). You declare
parts with descriptor decorators from `saeco.architecture`; the framework
introspects them to assemble the trainable, wire losses, enumerate sweeps,
resample, and save/load — no glue code:

- `@model_prop` — the one method that builds the core `SAE` (a composed `cl.Seq`)
- `@loss_prop` — a training loss; weighted by method name via `train_cfg.coeffs`
- `@aux_model_prop` — a secondary `SAE` with its own losses
- `setup()` — attaches layer mixins (e.g. `ft.NormFeaturesMixin`,
  `ft.OrthogonalizeFeatureGradsMixin`) that participate in the training loop

`self.cfg` is the arch config; `self.init` is the `Initializer` (parameter
factories sized from `init_cfg`). `RunConfig[ConfigT]` bundles
`train_cfg` / `init_cfg` / `resampler_config` / `normalizer_cfg`. The public
API is curated in `src/saeco/__init__.py` (`import saeco; saeco.Architecture`).
Rationale for these choices: `docs/architecture.md`.

## Sweepable DSL

`SweepableConfig` (a pydantic model) widens every field's type `T` to
`T | Swept[T] | SweepExpression[T]`. Drop `Swept(a, b, c)`, `SweepVar(...,
name=...)`, or `Val(value=...)`-based expressions anywhere a value is
expected, at any nesting depth. `is_concrete()` reports whether sweep axes
remain; `sweep_info_tree` (a cached property) describes the grid;
`select_instance_by_index(i)` / `random_sweep_configuration()` materialize
concrete configs. A `SweepVar` is one named axis shared across fields (couples
them instead of forming a cross-product). Type checkers flag `Swept[T]` as not
assignable to `T` — this is expected (runtime-handled).

## Design philosophy

The aim is for this infrastructure to stay light, malleable, and genuinely
fun to build on — where good abstractions compose so cleanly that hard
things turn out easy and you surprise yourself with how little code a
powerful change took.

The strictness here is *in service of* that lightness, not opposed to it.
Strong typing, avoiding stringly-typed indirection, and failing fast and
loudly when an expectation breaks are what keep the subtle-bug rate near
zero — and that is what makes fast, confident iteration possible. The loop
it buys you: write the ambitious code you're ~80% sure of that feels just
within reach, run it; if it works, the remaining uncertainty is mostly
resolved "yes, my mental model was right"; if it hard-errors, the error
usually points straight at the place your mental model was off. Lean into
that loop, and prefer a loud failure at the boundary over a silent
degradation, so the system keeps giving you this loud feedback from
reality.

This extends to how the work itself gets done. The same honesty the code
keeps at its boundaries, keep at yours: when there's a real fork, surface
it instead of quietly choosing for someone; let "done" mean done, not a
green check with the failures hidden or a status claimed past what's true.
This isn't a guard against you — it's the bargain that makes the fast,
trusting loop above possible, and you're now one of the people keeping it.
The conventions below — and the patterns you'll meet throughout the
codebase — are expressions of the same stance, not bureaucracy: typed
models (pydantic/attrs) whose validation is a fail-fast boundary, factory
methods and cached-property assembly, a bias toward immutability, and,
where it earns its keep, intricate machinery hidden behind a small,
elegant interface. Treat them as deliberate and extend them rather than
introducing a foreign style — and pass the codebase on in the same good
faith you found it in.

## Conventions

- `research/` is per-author scratch; promote a module to the library by
  `git mv`-ing it into `src/saeco/`.
- Don't restate a symbol's name as a string literal when the code already
  has a handle to that symbol — derive it (e.g. `descriptor.attrname`,
  `Type.__name__`). String names are invisible to renames and refactors.

## Gotchas

- **`nnsight` has a flaky `exec()`-based import** that seems to nondeterministically fail some of the time
  (`TypeError: exec() arg 1 ...`). It is not caused by repo code — retry
  `import saeco` / the test command if it errors at the nnsight import.
- Moving/renaming files across the `sweepable`/`saeco`/`research` boundary
  rewrites many import sites. Use `git mv` (preserves history) + a scoped
  grep/sed import rewrite; the LSP tool's `findReferences` can cross-check
  call sites but cannot perform the move/rename itself. Verify all three
  test suites still collect afterward.
