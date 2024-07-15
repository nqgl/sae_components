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
from saeco.trainer.runner import SAEConfig
from saeco.configs import RunSchedulingConfig


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


from saeco.trainer.runner import (
    TrainingRunner,
    TrainConfig,
    RunConfig,
)
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import DataConfig, ModelConfig, ActsDataConfig


model_fn = sae
quick_check = False
PROJECT = "sae sweeps"
train_cfg = TrainConfig(
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=not quick_check))
    ),
    l0_target=25,
    coeffs={
        "sparsity_loss": 4e-3,
        "L2_loss": 1,
    },
    lr=1e-3,
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0003,
    batch_size=4096,
    use_lars=True,
    betas=(0.9, 0.999),
    raw_schedule_cfg=RunSchedulingConfig(
        run_length=1000,
        lr_cooldown_length=0.2,
        lr_resample_warmup_length=0.3,
        lr_resample_warmup_factor=0.2,
        # targeting_resample_cooldown_period_override=(2_000),
        targeting_post_resample_cooldown=0.3,
        targeting_post_resample_hiatus=0.4,
        resample_period=(4_000),
        resampling_finished_phase=0.3,
        resample_delay=(5_000),
    ),
)
acfg = Config(
    pre_bias=True,
)
cfg = RunConfig[Config](
    train_cfg=train_cfg,
    arch_cfg=acfg,
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
    ),
    sae_cfg=SAEConfig(dict_mult=16),
)
tr = TrainingRunner(cfg, model_fn=model_fn)
data = tr.buf
model = tr.trainable
import torch

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
for i in range(1000):
    x = next(data)
    y = model(x)
    l = (y - x).pow(2).mean()
    l.backward()
    optim.step()
    optim.zero_grad()


def run(cfg):

    tr.trainer.train()
