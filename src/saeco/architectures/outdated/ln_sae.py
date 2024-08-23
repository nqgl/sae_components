import torch.nn as nn

from saeco.initializer import Initializer
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
from saeco.trainer.TrainConfig import TrainConfig


class Config(SweepableConfig):
    pre_bias: bool = False
    untied: bool = True
    d_extra: int = 5
    use_ln: bool = True


def ln_sae(
    init: Initializer,
    cfg: Config,
):
    init._encoder.add_wrapper(ReuseForward)
    if cfg.untied:
        init._decoder.add_wrapper(ft.NormFeatures)
        init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
    else:
        init._decoder.tie_weights(init._encoder)
    init._decoder._weight_tie = None
    init._encoder.d_in += cfg.d_extra

    def model(enc, penalties, metrics, detach=False):
        # return Seq(
        #     **useif(cfg.pre_bias % 3 == 1, pre_bias=init._decoder.sub_bias()),
        #     **useif(cfg.pre_bias < 3, ln=nn.LayerNorm(init.d_data)),
        #     **useif(cfg.pre_bias % 3 == 2, pre_bias=init._decoder.sub_bias()),
        #     encoder=enc,
        #     freqs=EMAFreqTracker(),
        #     **penalties,
        #     metrics=metrics,
        #     decoder=init.decoder,
        # )
        return Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            proj_extra=nn.Linear(init.d_data, init.d_data + cfg.d_extra),
            **useif(cfg.use_ln, ln=nn.LayerNorm(init.d_data + cfg.d_extra)),
            # **useif(cfg.pre_bias % 3 == 2, pre_bias=init._decoder.sub_bias()),
            encoder=enc,
            freqs=EMAFreqTracker(),
            **penalties,
            metrics=metrics,
            decoder=init.decoder,
        )

    model_full = model(
        enc=Seq(linear=init.encoder, relu=nn.ReLU()),
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


from saeco.trainer.RunConfig import RunConfig
from saeco.trainer.runner import TrainingRunner, DataConfig
from saeco.data.dataset import ModelConfig, ActsDataConfig
from saeco.sweeps import Swept, do_sweep


model_fn = ln_sae

PROJECT = "sae sweeps"
train_cfg = TrainConfig(
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=False))
    ),
    l0_target=45,
    coeffs={
        "sparsity_loss": 3e-4,
        "L2_loss": 1,
    },
    lr=7e-4,
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.001,
    batch_size=4096,
    use_lars=True,
    betas=Swept[tuple[float, float]]((0.9, 0.99)),
)
acfg = Config(
    pre_bias=Swept[bool](True, False),
    use_ln=Swept[bool](True, False),
    d_extra=Swept[int](0, 2, 5),
)
cfg = RunConfig[Config](
    train_cfg=train_cfg,
    arch_cfg=acfg,
)


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=model_fn)
    tr.trainer.train()


if __name__ == "__main__":

    do_sweep(True)
