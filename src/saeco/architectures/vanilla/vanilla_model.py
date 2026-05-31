import torch.nn as nn

import saeco.components.features as ft
from saeco.architecture import (
    component_prop,
    loss_prop,
    model_prop,
)
from saeco.architecture.sae import SAE
from saeco.architecture.sae_architecture import Architecture
from saeco.components import (
    L2Loss,
    SparsityPenaltyLoss,
)
from saeco.core import Seq
from saeco.misc import useif
from sweepable import SweepableConfig


class VanillaConfig(SweepableConfig):
    # SweepableConfig is a subclass of pydantic BaseModel
    pre_bias: bool = False
    # this is implicitly bool | Swept | SweepExpression due to being a SweepableConfig


class VanillaSAE(Architecture[VanillaConfig]):
    def setup(self):
        # Nothing to configure: the vanilla SAE needs no stateful setup of
        # self.init. The feature wrappers are applied explicitly where the
        # decoder is composed (see `decoder` below), not registered here.
        pass

    @component_prop
    def decoder(self):
        # These wrappers ensure:
        # 1. the features are normalized to unit norm after each optimizer step
        # 2. the feature gradients are orthogonalized after each backward pass,
        #    before the optimizer step.
        return ft.OrthogonalizeFeatureGrads(ft.NormFeatures(self.init.decoder))

    # model_prop tells the Architecture class that this method
    # is the method that constructs the model.
    # model_prop is a subclass of cached_property, so self.model will always
    # refer to the same instance of the model
    @model_prop
    def model(self):
        return SAE(
            encoder_pre=Seq(
                **useif(self.cfg.pre_bias, pre_bias=self.init.decoder.sub_bias),
                lin=self.init.encoder,
            ),
            nonlinearity=nn.ReLU(),
            decoder=self.decoder,
        )

    # loss_prop designates a Loss that will be used in training
    @loss_prop
    def L2_loss(self):  # noqa: N802  # loss-prop method name is the train_cfg.coeffs key
        return L2Loss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)
