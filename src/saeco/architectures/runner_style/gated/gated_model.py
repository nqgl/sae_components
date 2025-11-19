import torch
import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.components import (
    EMAFreqTracker,
    L1Penalty,
    L2Loss,
    SparsityPenaltyLoss,
)
from saeco.core import ReuseForward, Seq
from saeco.initializer import Initializer
from saeco.misc import useif
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.Swept import Swept


class Config(SweepableConfig):
    pre_bias: bool = Swept(True, False)
    detach: bool = Swept(True, False)
    # decay_l1_penalty: float =


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
        encoder=enc_gate,  # oh and it's missing 1-2 detatches
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
    pass
