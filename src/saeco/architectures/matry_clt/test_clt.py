from functools import cached_property

import saeco.components.features.features as ft

import saeco.core as cl
import torch
import torch.nn as nn

from saeco.architecture import Architecture, aux_model_prop, loss_prop, model_prop, SAE

from saeco.components import L2Loss, Lambda, Loss, SparsityPenaltyLoss
from saeco.components.sae_cache import SAECache
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward
from saeco.misc import useif
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig


class CrossLayerTranscoderConfig(SweepableConfig):
    pre_bias: bool = False  # placeholder until more config options are needed
    number_of_sites: int = 2


class CrossLayerTranscoder(Architecture[CrossLayerTranscoderConfig]):
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
    @cached_property
    def encode_each_layer(self):
        assert self.init.d_dict % self.cfg.number_of_sites == 0
        assert self.init.d_data % self.cfg.number_of_sites == 0
        d_layer_dict = self.init.d_dict // self.cfg.number_of_sites
        d_layer_data = self.init.d_data // self.cfg.number_of_sites

        def index_chunk(i):
            return Lambda(lambda x: x[:, d_layer_data * i : d_layer_data * (i + 1)])

        return cl.Parallel(
            *[
                Seq(
                    index_chunk(i),
                    nn.Linear(
                        in_features=d_layer_data,
                        out_features=d_layer_dict,
                        bias=False,
                    ),
                )
                for i in range(self.cfg.number_of_sites)
            ]
        ).reduce(lambda *x: torch.cat(x, dim=1))

    @model_prop
    def model(self):
        def index_up_to(i):
            return Lambda(lambda x: x[:, : self.init.d_dict * (i + 1)])

        return SAE(
            encoder=self.encode_each_layer,
            decoder=cl.Parallel(
                *[
                    Seq(
                        index_up_to(i),
                        nn.Linear(
                            in_features=self.init.d_dict * (i + 1),
                            out_features=self.init.d_data // self.cfg.number_of_sites,
                            bias=False,
                        ),
                    )
                    for i in range(self.cfg.number_of_sites)
                ]
            ).reduce(lambda *x: torch.cat(x, dim=1)),
        )

    @loss_prop
    def l2_loss(self):
        return L2Loss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)
