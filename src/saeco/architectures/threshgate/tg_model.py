import torch

import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.core import ReuseForward
from saeco.architectures.initialization.initializer import Initializer
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)

from saeco.components.hooks.clipgrad import ClipGrad
from saeco.core import Seq

from saeco.misc import useif
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.components.penalties.l1_penalizer import L0TargetingL1Penalty
from saeco.sweeps import SweepableConfig, Swept
import einops
from saeco.components.hierarchical import hl2ll


class ThreshGate(cl.Module):
    def __init__(self, p_soft: float, hard_cb, full_cb):
        super().__init__()
        self.hard_cb = hard_cb
        self.full_cb = full_cb
        self.p_soft = p_soft

    def forward(self, acts, gate_acts: torch.Tensor, *, cache: cl.Cache):
        # acts = acts.relu()
        hard = cache(self).hard(gate_acts, acts)
        cache(self).hard_cb(hard)
        if self.training:
            soft = cache(self).soft(gate_acts, acts)
            rand = torch.rand_like(gate_acts) < self.p_soft
            full = torch.where(rand, soft, hard)
            cache(self).full_cb(full)
            return full
        cache(self).full_cb(hard)
        return hard

    def hard(self, gate_acts, acts):
        return acts * (gate_acts > 0)

    def soft(self, gate_acts, acts):
        return acts * torch.sigmoid(gate_acts)


class Config(SweepableConfig):
    pre_bias: bool = Swept(True, False)
    detach: bool = Swept(True, False)
    bf: int = 32
    num_layers: int = 3
    l0_target_ratio: float = 2

    # decay_l1_penalty: float =


class Gate(cl.Module):
    def __init__(self, init: Initializer, cfg: Config):
        super().__init__()
        self.init = init
        self.cfg = cfg
        self.bias = init.new_encoder_bias()
        self.decoder = init.decoder
        self.encoder_bias = init.new_encoder_bias()
        self.decoder_bias = init.new_decoder_bias()
        encoder = init.encoder
        enc_weight = encoder.weight
        totsize = 0
        l0_targets = cfg.l0_target_ratio ** torch.arange(cfg.num_layers)
        self.layer_bounds = []
        for i in range(cfg.num_layers):
            bf = cfg.bf**i
            assert init.d_dict % bf == 0
            layer_size = init.d_dict // bf
            self.layer_bounds.append((slice(totsize, totsize + layer_size), bf))
            totsize += layer_size

        layers = [enc_weight]
        full_enc = nn.Linear(init.d_dict, totsize, bias=True)

        encoder.weight.data = full_enc.weight.data[self.layer_bounds[0][0]]
        self.encoder_0 = encoder
        self.encoder = full_enc

        self.threshgates = [
            ThreshGate(
                0.5,
                hard_cb=(penalty := L0TargetingL1Penalty(target=t).update_l0),
                full_cb=penalty,
            )
            for t in l0_targets
        ]

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        full_pre_acts = cache(self).encoder(x)
        prev_acts = None
        for i in reversed(range(self.cfg.num_layers - 1)):
            sl, bf = self.layer_bounds[i]
            pre_acts = full_pre_acts[sl]
            if prev_acts is None:
                acts = pre_acts
            else:
                acts = cache(self).threshgates[i](
                    pre_acts, hl2ll(prev_acts, self.cfg.bf)
                )
            prev_acts = acts
        return acts.relu()


def gated_sae(
    init: Initializer,
    cfg: Config,
):

    init._encoder.bias = False
    init._encoder.add_wrapper(ReuseForward)
    init._decoder.add_wrapper(ft.NormFeatures)
    init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
    enc_mag = Seq(
        **useif(
            cfg.pre_bias,
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

    enc_gate = ReuseForward(
        Seq(
            **useif(
                cfg.pre_bias,
                pre_bias=(
                    cl.Parallel(left=cl.ops.Identity(), right=init.decoder.bias).reduce(
                        (lambda l, r: l - r.detach())
                    )
                    if cfg.detach
                    else ReuseForward(init._decoder.sub_bias())
                ),
            ),
            weight=ReuseForward(init.encoder),
            bias=init.new_encoder_bias(),
            nonlinearity=nn.ReLU(),
        )
    )

    # models
    gated_model = Seq(
        encoder=cl.Parallel(
            magnitude=enc_mag,
            gate=co.ops.Thresh(enc_gate),
        ).reduce(
            lambda x, y: x * y,
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        decoder=init.decoder,
    )

    model_aux = Seq(  # this one is just used for training the gate appropriately
        encoder=enc_gate,  # oh and it's missing 1-2 detaches
        L1=L1Penalty(),
        freqs=EMAFreqTracker(),
        decoder=init._decoder.detached if cfg.detach else init.decoder,
    )

    # losses
    losses = dict(
        L2_loss=L2Loss(gated_model),
        L2_aux_loss=L2Loss(model_aux),
        sparsity_loss=SparsityPenaltyLoss(model_aux),
    )
    return [gated_model, model_aux], losses


from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=gated_sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    from .tg_config import cfg, PROJECT
