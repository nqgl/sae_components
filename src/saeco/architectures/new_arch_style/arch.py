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


class GatedConfig(SweepableConfig):
    pre_bias: bool = True
    detach: bool = True
    # decay_l1_penalty: float =


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
            encoder=cl.Parallel(
                magnitude=self.enc_mag,
                gate=co.ops.Thresh(self.enc_gate),
            ).reduce(
                lambda x, y: x * y,
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
