import torch.nn as nn

from saeco.architectures.initialization.initializer import Initializer
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)

from saeco.core.reused_forward import ReuseForward
from saeco.core import Seq
import saeco.components.features.features as ft

import saeco.components as co
from saeco.misc import useif
from saeco.sweeps import SweepableConfig


class Config(SweepableConfig):
    pre_bias: bool = False
    untied: bool = True


def sae(
    init: Initializer,
    cfg: Config,
):
    # init._encoder.add_wrapper(ReuseForward)
    if cfg.untied:
        init._decoder.add_wrapper(ft.NormFeatures)
        init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
    else:
        init._decoder.tie_weights(init._encoder)

    def model(enc, penalties, metrics, detach=False):
        return Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            encoder=enc,
            freqs=EMAFreqTracker(),
            **penalties,
            metrics=metrics,
            decoder=init.decoder.resampled(),
        )

    model_full = model(
        enc=Seq(linear=init.encoder.resampled(), relu=nn.ReLU()),
        penalties=dict(l1=L1Penalty()),
        metrics=co.metrics.ActMetrics(),
        detach=False,
    )

    models = [model_full]
    losses = dict(
        L2_loss=L2Loss(model_full),
        sparsity_loss=SparsityPenaltyLoss(model_full),
    )

    return models, losses


from saeco.trainer.runner import TrainingRunner, TrainConfig, RunConfig
from saeco.data import DataConfig, ModelConfig, ActsDataConfig
from saeco.sweeps import Swept, do_sweep


model_fn = sae

PROJECT = "sae sweeps"
train_cfg = TrainConfig(
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=False))
    ),
    l0_target=None,
    coeffs={
        "sparsity_loss": 5e-3,
        "L2_loss": 1,
    },
    lr=7e-4,
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.001,
    batch_size=2048,
    use_lars=True,
    betas=Swept[tuple[float, float]]((0.9, 0.99)),
)
acfg = Config(
    pre_bias=Swept[bool](True, False),
)
cfg = RunConfig[Config](
    train_cfg=train_cfg,
    arch_cfg=acfg,
)


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=model_fn)
    tr.trainer.train()


if __name__ == "__main__":

    do_sweep(True, "rand")
