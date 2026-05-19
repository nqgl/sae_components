# saeco

[![CI](https://github.com/nqgl/sae_components/actions/workflows/ci.yml/badge.svg)](https://github.com/nqgl/sae_components/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

A modular library for training and analyzing sparse autoencoders (SAEs) for
mechanistic interpretability. You compose architectures from small building
blocks; the library handles training, data caching, resampling, L0 targeting,
sweeping, and remote orchestration.

## Why saeco?

- **Build architectures from small, named pieces.** Declare the model,
  losses, and aux losses for an SAE; the library handles the training
  pipeline, save/load, sweep enumeration, and resampling for you.
- **Sweeping is a first-class config feature.** Any field on a
  `SweepableConfig` can be a `Swept(a, b, c)` or a `SweepExpression(...)`
  — the trainer enumerates the combinations for you.
- **Composable layers.** Mixins like `NormFeaturesMixin` and
  `OrthogonalizeFeatureGradsMixin` attach to layers declaratively and
  participate in the standard training loop.

See [docs/architecture.md](docs/architecture.md) for the rationale behind these choices.

## Installation

`saeco` depends on the [`sweepable`](sweepable/) package which lives in
this same repo and is not on PyPI yet. From the repo root, install the
dependency chain in editable mode:

```bash
python -m pip install -e ./sweepable
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[remote]"     # remote sweep orchestration (ezpod)
python -m pip install -e ".[dev]"        # pytest, ruff, pre-commit
```

## Quickstart: Vanilla SAE

The full reference implementation lives at
[`src/saeco/architectures/vanilla/vanilla_model.py`](src/saeco/architectures/vanilla/vanilla_model.py).

```python
import torch.nn as nn

import saeco.components.features as ft
from saeco import SAE, Architecture, SweepableConfig, loss_prop, model_prop
from saeco.components import L2Loss, SparsityPenaltyLoss
from saeco.core import Seq
from saeco.misc import useif


class VanillaConfig(SweepableConfig):
    # SweepableConfig is a subclass of pydantic BaseModel.
    # Each field is implicitly `T | Swept[T] | SweepExpression`.
    pre_bias: bool = False


class VanillaSAE(Architecture[VanillaConfig]):
    def setup(self):
        # Mixins added during setup participate in the training loop:
        # 1. features are normalized to unit norm after each optimizer step
        # 2. feature gradients are orthogonalized before each step
        self.init._decoder.add_mixin_(ft.NormFeaturesMixin)
        self.init._decoder.add_mixin_(ft.OrthogonalizeFeatureGradsMixin)

    @model_prop
    def model(self):
        return SAE(
            encoder_pre=Seq(
                **useif(self.cfg.pre_bias, pre_bias=self.init._decoder.sub_bias()),
                lin=self.init.encoder,
            ),
            nonlinearity=nn.ReLU(),
            decoder=self.init.decoder,
        )

    @loss_prop
    def L2_loss(self):
        return L2Loss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)
```

For a runnable end-to-end script that wires this up to a `RunConfig`, a
small sweep grid, and a training run, see
[`examples/train_vanilla_sae.py`](examples/train_vanilla_sae.py). Two more
standalone demos live alongside it:
[`examples/define_architecture.py`](examples/define_architecture.py) and
[`examples/sweep_dsl.py`](examples/sweep_dsl.py).

## A more involved architecture: Gated SAE

The Gated SAE shows the rest of the library: parallel paths, auxiliary
models with their own losses, conditional setup based on config flags, and
detached gradients across paths.

```python
from functools import cached_property

import torch
import torch.nn as nn

import saeco.components as co
import saeco.components.features as ft
import saeco.core as cl
from saeco import SAE, Architecture, SweepableConfig, aux_model_prop, model_prop
from saeco.components import EMAFreqTracker, L2Loss, SparsityPenaltyLoss
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward
from saeco.misc import useif


class GatedConfig(SweepableConfig):
    pre_bias: bool = False
    detach: bool = True


class Gated(Architecture[GatedConfig]):
    def setup(self):
        self.init._encoder.bias = False
        self.init._encoder.add_wrapper(ReuseForward)
        self.init._decoder.add_mixin_(ft.NormFeaturesMixin)
        self.init._decoder.add_mixin_(ft.OrthogonalizeFeatureGradsMixin)

    @cached_property
    def enc_mag(self):
        return Seq(
            **useif(
                self.cfg.pre_bias,
                pre_bias=ReuseForward(self.init._decoder.sub_bias()),
            ),
            r_mag=cl.ops.MulParallel(
                identity=ReuseForward(self.init.encoder),
                exp_r=co.Lambda(
                    func=lambda x: torch.exp(x),
                    module=self.init.dict_bias(),
                ),
            ),
            bias=self.init.new_encoder_bias(),
            nonlinearity=nn.ReLU(),
        )

    @cached_property
    def enc_gate(self):
        return Seq(
            **useif(
                self.cfg.pre_bias,
                pre_bias=(
                    cl.Parallel(
                        left=cl.ops.Identity(), right=self.init.decoder.bias
                    ).reduce((lambda l, r: l - r.detach()))
                    if self.cfg.detach
                    else ReuseForward(self.init._decoder.sub_bias())
                ),
            ),
            weight=ReuseForward(self.init.encoder),
            bias=self.init.new_encoder_bias(),
            nonlinearity=nn.ReLU(),
        )

    @model_prop
    def gated_model(self):
        return SAE(
            encoder=cl.Parallel(
                magnitude=self.enc_mag,
                gate=co.ops.Thresh(self.enc_gate),
            ).reduce(lambda x, y: x * y),
            decoder=self.init.decoder,
            penalty=None,
        )

    L2_loss = gated_model.add_loss(L2Loss)

    @aux_model_prop
    def model_aux(self):
        return SAE(
            encoder=self.enc_gate,
            freqs=EMAFreqTracker(),
            decoder=(
                self.init._decoder.detached if self.cfg.detach else self.init.decoder
            ),
        )

    L2_aux_loss = model_aux.add_loss(L2Loss)
    sparsity_loss = model_aux.add_loss(SparsityPenaltyLoss)
```

## Repository layout

```
src/saeco/                  # the library — public API
├── architecture/           # Architecture, SAE, *_prop decorators
├── architectures/          # reference architectures (vanilla, gated)
├── components/             # losses, penalties, ops, features, resampling
├── core/                   # Seq, Parallel, MulParallel, Cache, …
├── data/                   # dataset config, tokenization, activation caches
├── initializer/            # parameter initialization (incl. geometric median)
├── mlog/                   # logging (currently W&B-backed)
├── sweeps/                 # do_sweep + thin re-exports of sweepable's DSL
└── trainer/                # Trainer, RunConfig, scheduling, normalizers

sweepable/                  # standalone subrepo — Pydantic + sweep DSL
├── pyproject.toml          # installs as `sweepable` (no saeco deps)
├── src/sweepable/          # SweepableConfig, Swept, SweepVar, Val, …
└── tests/

research/                   # experimental extension package
                            # installs as `saeco-research`; no API guarantees

examples/                   # curated standalone demos
tests/                      # library tests
```

This repo is a "monorepo of three packages" (`sweepable` → `saeco` →
`saeco-research`). To use them all:

```bash
python -m pip install -e ./sweepable    # the sweep DSL (used by saeco)
python -m pip install -e .              # the library
python -m pip install -e ./research     # the research extension package
```

## Status

This is an active research-driven project. The library is stable enough to
build on, and the reference architectures (`vanilla`, `gated`) work
end-to-end. Logging is W&B-backed by default in this iteration, with calls
centralized through `saeco.mlog`.

Feel free to open an issue or contact me if you have questions.

## License

Apache 2.0 — see [LICENSE](LICENSE).
