import torch
import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.components import (
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)
from saeco.core import Seq
from saeco.initializer import Initializer
from saeco.misc import useif
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.Swept import Swept


class Config(SweepableConfig):
    pre_bias: bool = False
    relu_first_acts: bool = Swept(True, False)
    l1_1_scale: float = Swept(0, 0.1, 0.5, 1)
    gelu_mid: bool = Swept(True, False)


def sae(
    init: Initializer,
    cfg: Config,
):
    dg = nn.Linear(init.d_dict, init.d_dict, bias=True)
    dg.weight.data = torch.eye(init.d_dict)
    model = Seq(
        encoder=Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            lin=init.encoder,
        ),
        penalty1=Seq(nn.ReLU(), co.L1Penalty(scale=cfg.l1_1_scale)),
        dg=cl.Parallel(
            Seq(nn.GELU() if cfg.gelu_mid else nn.ReLU(), dg),
            nn.ReLU() if cfg.relu_first_acts else cl.ops.Identity(),
        ).reduce(lambda a, b: (a + b) / 2),
        relu=nn.ReLU(),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        penalty2=co.L1Penalty(),
        decoder=ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                init.decoder,
            ),
        ),
    )

    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model, num_expeted=2),
    )
    return models, losses


from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    pass
