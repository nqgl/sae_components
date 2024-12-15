from functools import cached_property
import torch
from saeco.architecture.architecture import Architecture, SAE
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


class MyConfig(SweepableConfig):
    abool: bool
    pre_bias: bool = False


class MyArch(Architecture[MyConfig]):

    def build_pre_encoder(self):
        return Seq(
            **useif(self.cfg.pre_bias, pre_bias=self.init._decoder.sub_bias()),
            lin=self.init.encoder,
        )

    def build_nonlinearity(self):
        return nn.ReLU()

    def build_encoder(self):
        return Seq(
            **useif(self.cfg.pre_bias, pre_bias=self.init._decoder.sub_bias()),
            lin=self.init.encoder,
            nonlinearity=nn.ReLU(),
        )


# class VanillaSAE(Architecture[MyConfig]):
#     def make_encoder_pre(self):
#         return Seq(
#             **useif(self.cfg.pre_bias, pre_bias=self.init._decoder.sub_bias()),
#             lin=self.init.encoder,
#         )

#     def make_nonlinearity(self):
#         return nn.ReLU()

#     def decoder_pre(self):
#         return ft.OrthogonalizeFeatureGrads(
#             ft.NormFeatures(
#                 self.init.decoder,
#             ),
#         )

#     models = [model]
#     losses = dict(
#         L2_loss=L2Loss(model),
#         sparsity_loss=SparsityPenaltyLoss(model),
#     )
#     # return models, losses


class VanillaSAE(Architecture[MyConfig]):
    def build(self):
        self.encoder_pre = Seq(
            **useif(self.cfg.pre_bias, pre_bias=self.init._decoder.sub_bias()),
            lin=self.init.encoder,
        )

        self.nonlinearity = nn.ReLU()

        self.decoder = ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                self.init.decoder,
            ),
        )


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
        Seq(
            **useif(
                self.cfg.pre_bias,
                pre_bias=ReuseForward(init._decoder.sub_bias()),
            ),
            r_mag=cl.ops.MulParallel(
                identity=ReuseForward(init.encoder),
                exp_r=co.Lambda(
                    func=lambda x: torch.exp(x),
                    module=init.dict_bias(),
                ),
            ),
            bias=init.new_encoder_bias(),
            nonlinearity=nn.ReLU(),
        )

    @cached_property
    def enc_gate(self):
        return Seq(
            **useif(
                self.cfg.pre_bias,
                pre_bias=(
                    cl.Parallel(left=cl.ops.Identity(), right=init.decoder.bias).reduce(
                        (lambda l, r: l - r.detach())
                    )
                    if self.cfg.detach
                    else ReuseForward(init._decoder.sub_bias())
                ),
            ),
            weight=ReuseForward(init.encoder),
            bias=init.new_encoder_bias(),
            nonlinearity=nn.ReLU(),
        )

    @cached_property
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

    @cached_property
    def model_aux(self):
        return SAE(  # this one is just used for training the gate appropriately
            encoder=self.enc_gate,  # oh and it's missing 1-2 detaches
            freqs=EMAFreqTracker(),
            decoder=(
                self.init._decoder.detached if self.cfg.detach else self.init.decoder
            ),
        )

    # losses
    losses = dict(
        L2_loss=L2Loss(gated_model),
        L2_aux_loss=L2Loss(model_aux),
        sparsity_loss=SparsityPenaltyLoss(model_aux),
    )
    # return [gated_model, model_aux], losses

    def make_encoder_pre(self):
        return Seq(
            **useif(self.cfg.pre_bias, pre_bias=self.init._decoder.sub_bias()),
            lin=self.init.encoder,
        )

    def make_nonlinearity(self):
        return nn.ReLU()

    def decoder_pre(self):
        return ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                self.init.decoder,
            ),
        )


class Gated(Architecture[GatedConfig]):
    def build(self):
        assert self.primary_model is None
        assert self.aux_models is None
        # this should be in _build or something in the parent class

        self.init._encoder.bias = False
        self.init._encoder.add_wrapper(ReuseForward)
        self.init._decoder.add_wrapper(ft.NormFeatures)
        self.init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)

        enc_mag = Seq(
            **useif(
                cfg.pre_bias,
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

        enc_gate = ReuseForward(
            Seq(
                **useif(
                    cfg.pre_bias,
                    pre_bias=(
                        cl.Parallel(
                            left=cl.ops.Identity(), right=self.init.decoder.bias
                        ).reduce((lambda l, r: l - r.detach()))
                        if cfg.detach
                        else ReuseForward(self.init._decoder.sub_bias())
                    ),
                ),
                weight=ReuseForward(self.init.encoder),
                bias=self.init.new_encoder_bias(),
                nonlinearity=nn.ReLU(),
            )
        )


Model[GatedConfig](
    encoder_pre=enc_mag,
    nonlinearity=cl.Parallel(
        magnitude=co.Identity(),
        gate=co.ops.Thresh(enc_gate),
    ).reduce(
        lambda x, y: x * y,
    ),
    decoder=init.decoder,
)
# class GatedModel(Model[GatedConfig]):
#     @classmethod
#     def build(cls, init, cfg, )
