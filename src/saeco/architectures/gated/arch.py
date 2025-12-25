from functools import cached_property

import torch
import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.architecture import (
    SAE,
    Architecture,
    aux_model_prop,
    model_prop,
)
from saeco.components import (
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward
from saeco.misc import useif
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig


class GatedConfig(SweepableConfig):
    pre_bias: bool = False
    detach: bool = True


class Gated(Architecture[GatedConfig]):
    def setup(self):
        self.init._encoder.bias = False
        self.init._encoder.add_wrapper(ReuseForward)
        self.init._decoder.add_wrapper(ft.NormFeatures)
        self.init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)

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
            encoder=cl.collections.MulParallel(
                magnitude=self.enc_mag,
                gate=co.ops.Thresh(self.enc_gate),
            ),
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
