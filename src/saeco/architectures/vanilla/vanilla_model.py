import torch.nn as nn

import saeco.components.features.features as ft
from saeco.architecture import (
    SAE,
    Architecture,
    loss_prop,
    model_prop,
)
from saeco.components import (
    L2Loss,
    SparsityPenaltyLoss,
)
from saeco.core import Seq
from saeco.misc import useif
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig


class VanillaConfig(SweepableConfig):
    # SweepableConfig is a subclass of pydantic BaseModel
    pre_bias: bool = False
    # this is implicitly bool | Swept | SweepExpression due to being a SweepableConfig


class VanillaSAE(Architecture[VanillaConfig]):
    # setup is called before models are constructed
    def setup(self):
        # these will add wrappers to the decoder that ensure:
        # 1. the features are normalized after each optimizer step to have unit norm
        # 2. the gradients of the features are orthogonalized after each backward pass before the optimizer step
        self.init._decoder.add_wrapper(ft.NormFeatures)
        self.init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)

    # model_prop tells the Architecture class that this method
    # is the method that constructs the model.
    # model_prop is a subclass of cached_property, so self.model will always
    # refer to the same instance of the model
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

    # loss_prop designates a Loss that will be used in training
    @loss_prop
    def L2_loss(self):
        return L2Loss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)
