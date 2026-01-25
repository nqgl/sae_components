# port of sweep_tg/sthreshgrad.py


import saeco
import saeco.components as co
import saeco.components.hooks.feature_hooks
import saeco.core as cl
from saeco.architecture import SAE, Architecture, loss_prop, model_prop
from saeco.architectures.sweep_tg.orig_rewrite.threshgrad_v1 import (
    ThreshGate,
    ThreshGateConfig,
)
from saeco.components import EMAFreqTracker, L2Loss, SparsityPenaltyLoss
from saeco.core import Seq
from saeco.misc import useif
from saeco.sweeps import SweepableConfig

# useif
ThreshGateConfig


class TGSAEConfig(SweepableConfig):
    thresh_gate_cfg: ThreshGateConfig
    pre_bias: bool = False
    decay_l1_steps: int | None = None
    penalize_in_gate: bool = True
    penalize_after_gate: bool = False


class TGArch(Architecture[TGSAEConfig]):
    def setup(self):
        self.init._decoder.add_mixin_(
            saeco.components.hooks.feature_hooks.NormFeaturesMixin
        )
        self.init._decoder.add_mixin_(
            saeco.components.hooks.feature_hooks.OrthogonalizeFeatureGradsMixin
        )

    @model_prop
    def model(self):
        def make_penalty_fn(decay_n: int | None = None):
            return (
                co.LinearDecayL1Penalty(40_000)
                if decay_n is not None
                else co.L1Penalty()
            )

        init = self.init
        s = SAE(
            encoder=Seq(
                **useif(self.cfg.pre_bias, pre_bias=self.init._decoder.sub_bias()),
                lin=ThreshGate(
                    self.cfg.thresh_gate_cfg,
                    init=self.init,
                    penalty=(
                        make_penalty_fn(self.cfg.decay_l1_steps)
                        if self.cfg.penalize_in_gate
                        else None
                    ),
                ),
            ),
            freqs=EMAFreqTracker(),
            # metrics=co.metrics.ActMetrics(),
            penalty=(
                make_penalty_fn(self.cfg.decay_l1_steps)
                if self.cfg.penalize_after_gate
                else cl.ops.Identity()
            ),
            decoder=init.decoder,
        )
        return s

    @loss_prop
    def L2_loss(self):
        return L2Loss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)
