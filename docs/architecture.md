# Design notes

> Draft.

## The problem

Training sparse autoencoders for interpretability is mostly *not* the
autoencoder. A single experiment drags along: activation capture and
caching from a subject model, a training loop with resampling and
learning-rate scheduling, sparsity/L0 targeting, metric logging, and —
because nobody runs one config — sweep orchestration across many runs
and often many machines.

Most of that is identical across architectures. The interesting part —
"what is the encoder/decoder, and what are the losses?" — is small. saeco
is built around making that small part the only thing you write, and
making everything else fall out of it.

## Architectures are Python classes, not config files

An architecture is a class that declares its pieces with decorators:

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

**Why not config files?** Architectures are *programs*: a Gated SAE has
conditional structure (detached vs. non-detached paths), shared
sub-modules, and parallel branches. Expressing that in YAML reinvents a
worse Python. Keeping it as code means the full language — closures,
conditionals, composition — is available, while the decorators preserve
the introspectability that config files are usually chosen for.

The cost: there's framework magic (metaclass-ish field collection) a
reader has to learn once. The bet is that "learn one pattern, then every
architecture is ~30-300 lines" beats "no magic, but every architecture
re-implements the training wiring." Enabling fast research iteration
weights values towards speed of implementation and standardization (for the sake of easy comparison)
versus readability and low magic. # todo wording on this 

## Sweeping is a property of the config, not a separate system

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

**Why this instead of a separate sweep file/format?** The usual split —
a config plus a sweep YAML that references config paths by string —
means the sweep spec can silently drift from the config schema. Making
the sweep live *in* the config keeps it type-checked against the same
model, lets a `SweepVar` couple several fields onto one axis (e.g.
hold a compute budget fixed while varying batch size, not as a
cross-product), and means
"a sweep" and "a run" are the same type — `is_concrete()` is the only
distinction. The sweep machinery is also the lowest-dependency part of
the codebase, which is why it could be extracted (see below).

## Three packages, one repo

```
sweepable      pydantic + sweep DSL, zero saeco deps
   ↑
  saeco        the library: architectures, training, data
   ↑
saeco-research in-flight scratch: experimental archs, eval, analysis
```

`sweepable` has no saeco-specific dependencies and is independently
useful, so it's a standalone, separately-installable package. `saeco` is the polished, API-stable surface.
`saeco-research` is the unstable exploratory regime - experimental architectures, evaluation code, analysis GUIs. Code in here may get be staled or incomplete.

<!-- **Why the split?** It lets the library have a clearer API without
throwing away in-progress work or pretending everything is stable.
What's load-bearing (`src/saeco/`) versus a research notebook
(`research/`) is obvious at a glance. The
dependency arrows only point one way, so nothing in the stable layer
reaches into scratch code. The boundary is enforced mechanically:
packaging only ships `src/saeco`, CI lints/tests each layer, and a
tombstone module fails loudly if old import paths are used after a move. -->

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
  (`saeco.mlog`) exists and is being migrated to; the abstraction is
  intentionally thin until the second backend justifies more.
- **Python 3.13 only.** The architecture-config typing uses PEP 695
  generics; these are too nice to give up.

## Map

| Area | Where | Notes |
|---|---|---|
| Architecture base + decorators | `saeco.architecture` | `Architecture`, `SAE`, `*_prop` |
| Building blocks | `saeco.core`, `saeco.components` | `Seq`, `Parallel`, losses, penalties, mixins |
| Training | `saeco.trainer` | `Trainer`, `RunConfig`, schedule, normalizers |
| Data | `saeco.data` | dataset config, tokenization, activation caches |
| Sweep DSL | `sweepable` | standalone package |
| Scratch | `research/` | experimental archs, eval API, analysis |
