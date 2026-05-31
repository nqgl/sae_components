import torch
import torch.nn as nn

import saeco.components as co
import saeco.components.features as ft
import saeco.core as cl
from saeco.architecture import (
    aux_model_prop,
    component_prop,
    loss_prop,
    model_prop,
)
from saeco.architecture.sae import SAE
from saeco.architecture.sae_architecture import Architecture
from saeco.components import (
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward
from saeco.misc import useif
from sweepable import SweepableConfig


class GatedConfig(SweepableConfig):
    pre_bias: bool = False
    detach: bool = True


class Gated(Architecture[GatedConfig]):
    def setup(self):
        # Stateful configuration only: the gate/magnitude encoders share a
        # single bias built below, so the underlying encoder linear has none.
        self.init.enc_builder.bias = False

    @component_prop
    def encoder(self):
        # One ReuseForward-wrapped encoder, shared across the magnitude and
        # gate paths so its forward is computed once per pass.
        return ReuseForward(self.init.encoder)

    @component_prop
    def decoder(self):
        return ft.OrthogonalizeFeatureGrads(ft.NormFeatures(self.init.decoder))

    @component_prop
    def enc_mag(self):
        return Seq(
            **useif(
                self.cfg.pre_bias,
                pre_bias=ReuseForward(self.init.decoder.sub_bias),
            ),
            r_mag=cl.ops.MulParallel(
                identity=self.encoder,
                exp_r=co.Lambda(
                    func=lambda x: torch.exp(x),
                    module=self.init.dict_bias(),
                ),
            ),
            bias=self.init.new_encoder_bias(),
            nonlinearity=nn.ReLU(),
        )

    @component_prop
    def enc_gate(self):
        return Seq(
            **useif(
                self.cfg.pre_bias,
                pre_bias=(
                    cl.Parallel(
                        left=cl.ops.Identity(), right=self.init.decoder.bias
                    ).reduce((lambda l, r: l - r.detach()))
                    if self.cfg.detach
                    else ReuseForward(self.init.decoder.sub_bias)
                ),
            ),
            weight=self.encoder,
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
            decoder=self.decoder,
            penalty=None,
        )

    @loss_prop
    def L2_loss(self):  # noqa: N802  # loss-prop method name is the train_cfg.coeffs key
        return L2Loss(self.gated_model)

    @aux_model_prop
    def model_aux(self):
        return SAE(
            encoder=self.enc_gate,
            freqs=EMAFreqTracker(),
            decoder=(
                self.init.decoder.as_detached() if self.cfg.detach else self.decoder
            ),
        )

    @loss_prop
    def L2_aux_loss(self):  # noqa: N802  # loss-prop method name is the train_cfg.coeffs key
        return L2Loss(self.model_aux)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model_aux)
