from functools import cached_property
import torch
from saeco.core.reused_forward import ReuseForward
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig
from saeco.core import Seq

from saeco.misc import useif
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)
import saeco.components.features.features as ft
import torch.nn as nn
import saeco.core as cl
import saeco.components as co

from saeco.architecture import (
    loss_prop,
    model_prop,
    aux_model_prop,
    Architecture,
    SAE,
)
from saeco.architecture.arch_prop import arch_prop
from types import GenericAlias


class Config(SweepableConfig):
    pre_bias: bool = False


class VanillaSAE(Architecture[Config]):
    def setup(self):
        self.init._decoder.add_wrapper(ft.NormFeatures)
        self.init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)

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

    L2_loss = model.add_loss(L2Loss)
    sparsity_loss = model.add_loss(SparsityPenaltyLoss)
