# Architecture Design

## The problem

Training sparse autoencoders for interpretability involves much more than
the autoencoder itself. A single experiment drags along activation capture
and caching from a subject model, a training loop with resampling and
learning-rate scheduling, sparsity/L0 targeting, metric logging, and
sweep orchestration across many runs and often many machines.

Most of that is identical across architectures. The interesting part —
"what is the encoder/decoder, and what are the losses?" — is small. saeco
is built around making that architecture-specific part the code you write
directly, while the common training, sweeping, saving, and evaluation
machinery is assembled from it.

## Architecture classes declare the trainable pieces

An architecture declares its model and losses as named properties:

```python
class VanillaSAE(Architecture[VanillaConfig]):
    def setup(self):
        self.init._decoder.add_mixin_(ft.NormFeaturesMixin)

    @model_prop
    def model(self):
        return SAE(encoder=self.init.encoder, decoder=self.init.decoder, ...)

    @loss_prop
    def L2_loss(self):
        return L2Loss(self.model)
```

The decorators (`model_prop`, `loss_prop`, `aux_model_prop`) are
`cached_property` subclasses that also *register* the method as a named,
discoverable field. The framework introspects those fields to assemble
the trainable, wire up losses (weighted by name from
`train_cfg.coeffs`), run aux models, and serialize/reload — without the
architecture author writing any of that glue.

This keeps architecture definitions compact while still making their
load-bearing parts discoverable by the framework. A Gated SAE can use
conditional structure, shared submodules, and parallel branches directly
in the model definition; the decorators give those pieces stable names
for training, logging, serialization, and comparison across runs.

The cost is that a reader has to learn a small set of framework
conventions. The benefit is that new architectures share a standard
shape: the model and losses are easy to find, while common training
glue stays in the library.

## Sweeps live inside the same config objects

Any field of a `SweepableConfig` can hold a concrete value, a
`Swept(a, b, c)`, or a `SweepExpression` built from `SweepVar`s/`Val`s.
The field's declared type `T` is implicitly widened to
`T | Swept[T] | SweepExpression[T]`, so a sweep can be dropped in
*anywhere* a value is expected, at any nesting depth:

```python
RunConfig[VanillaConfig](
    train_cfg=TrainConfig(lr=Swept(1e-3, 3e-4, 1e-4)),
    arch_cfg=VanillaConfig(pre_bias=Swept(True, False)),
)  # a 6-run grid, described in the same object as a single run
```

Keeping sweep values inside the config keeps them attached to the same
schema as single-run values. A `SweepVar` can couple several fields onto
one sweep axis, such as holding a compute budget fixed while varying batch size
and step count together. In practice, "a sweep" and "a run" are the same
type; `is_concrete()` is the distinction. The sweep machinery is also the
lowest-dependency part of the codebase, which is why it could be extracted
(see below) into a standalone package.

## Three packages, one repo

```
sweepable      pydantic + sweep DSL, zero saeco deps
   ↑
  saeco        the library: architectures, training, data
   ↑
saeco-research in-flight scratch: experimental archs, eval, analysis
```

`sweepable` has no saeco-specific dependencies and is independently
useful, so it is a standalone, separately-installable package. `saeco`
is the stable library surface. `saeco-research` contains experimental
architectures, evaluation code, analysis GUIs, and scripts that are useful
as working references but do not have API guarantees.

**Why the split?** It lets the library have a clear public API without
throwing away active research work or pretending every experiment is ready
for reuse. The important boundary is easy to see: `src/saeco/` is the
supported package; `research/` is an installable, unsupported extension
space. The dependency arrows point one way, so the stable layer does not
reach into research code.

## Composable layers via mixins

Behaviors like "renormalize features to unit norm after each optimizer
step" or "orthogonalize feature gradients before the step" aren't
properties of the *model* — they're properties of *training*. They're
attached declaratively in `setup()`:

```python
self.init._decoder.add_mixin_(ft.NormFeaturesMixin)
self.init._decoder.add_mixin_(ft.OrthogonalizeFeatureGradsMixin)
```

The mixin participates in the standard training loop at the right hook
points. This keeps the model definition about *structure* and pushes
*training-time behavior* into composable units that can be mixed onto
any layer, rather than special-cased in each architecture's loss or a
bespoke training loop.

## Things deliberately not done

- **No plugin/registry layer.** Architectures are imported and used
  directly; there's no string-keyed registry. Fewer indirections,
  easier to follow.
- **Logging is W&B-backed today.** A backend-agnostic logging layer
  (`saeco.mlog`) centralizes logging calls, but W&B remains the default
  backend in this iteration. The abstraction stays intentionally thin
  until another backend is actually needed.
- **Python 3.13+ only.** The architecture-config typing uses PEP 695
  generics; these are too nice to give up.

## Map

| Area | Where | Notes |
|---|---|---|
| Architecture base + decorators | `saeco.architecture` | `Architecture`, `SAE`, `*_prop` |
| Building blocks | `saeco.core`, `saeco.components` | `Seq`, `Parallel`, losses, penalties, mixins |
| Training | `saeco.trainer` | `Trainer`, `RunConfig`, schedule, normalizers |
| Data | `saeco.data` | dataset config, tokenization, activation caches |
| Sweep DSL | `sweepable` | standalone package |
| Research extensions | `research/` | experimental archs, eval API, analysis |
