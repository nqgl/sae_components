"""
Defining a custom SAE architecture
==================================

Architectures in saeco are written as small Python classes. You declare:

  - the config (a `SweepableConfig` subclass — every field is implicitly
    sweepable)
  - the model (via `@model_prop`)
  - one or more losses (via `@loss_prop`)
  - optional setup hooks that wrap layers (e.g. unit-norm features,
    orthogonalized feature gradients)

Below is a minimal-but-complete reimplementation of the vanilla SAE.
This example is intended to be *read*. It defines the architecture but
doesn't run a training loop — see `examples/train_vanilla_sae.py` for
that.
"""

import torch.nn as nn

import saeco.components.features as ft
from saeco import SAE, Architecture, SweepableConfig, loss_prop, model_prop
from saeco.components import L2Loss, SparsityPenaltyLoss
from saeco.core import Seq
from saeco.misc import useif


class MyVanillaConfig(SweepableConfig):
    """Each field is implicitly `T | Swept[T] | SweepExpression`."""

    pre_bias: bool = False


class MyVanillaSAE(Architecture[MyVanillaConfig]):
    def setup(self):
        # Mixins participate in the standard training loop:
        # - NormFeaturesMixin: re-normalize features to unit norm after each step
        # - OrthogonalizeFeatureGradsMixin: orthogonalize feature grads pre-step
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


if __name__ == "__main__":
    # The class itself is fully constructed at import time. To actually
    # *instantiate* an architecture you need a RunConfig (which bundles the
    # train config, resampler config, init config, and arch config). See
    # examples/train_vanilla_sae.py for the full pipeline.
    config_class = MyVanillaSAE.get_arch_config_class()
    print(f"Architecture class: {MyVanillaSAE.__name__}")
    print(f"  config class:     {getattr(config_class, '__name__', config_class)}")
    print(f"  config fields:    {list(MyVanillaConfig.model_fields)}")
