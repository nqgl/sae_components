import torch

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
from saeco.trainer.normalizers import GNConfig
import saeco.components as co
from saeco.misc import useif
from saeco.sweeps import SweepableConfig


class Config(SweepableConfig):
    pre_bias: bool = False
    untied: bool = True
    freq_balance_l0_target: int = 45


def sae(
    init: Initializer,
    cfg: Config,
):
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


from saeco.trainer import RunSchedulingConfig
from saeco.trainer.runner import TrainingRunner, TrainConfig, RunConfig
from saeco.data import DataConfig, ModelConfig, ActsDataConfig
from saeco.sweeps import Swept, do_sweep


model_fn = sae

PROJECT = "sae sweeps"
quick_check = False
train_cfg = TrainConfig(
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=not quick_check))
    ),
    raw_schedule_cfg=RunSchedulingConfig(
        run_length=50_000,
        resample_period=50_000,
        targeting_post_resample_hiatus=0,
        targeting_post_resample_cooldown=0.2,
        # resample_delay=0.69,
    ),
    l0_target=25,
    coeffs={
        "sparsity_loss": 4e-3,
        "L2_loss": 1,
    },
    lr=Swept(1e-3, 3e-4, 1e-4),
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0001,
    batch_size=4096,
    use_lars=True,
    betas=Swept[tuple[float, float]]((0.9, 0.999)),
)
acfg = Config(
    pre_bias=Swept[bool](True),
)
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)

runcfg = RunConfig[Config](
    # normalizer_cfg=GNConfig(
    #     mu_s=Swept(
    #         GNConfig.SAggregation.DONTUSE,
    #         GNConfig.SAggregation.PRIMED,
    #         GNConfig.SAggregation.SAMPLE,
    #     ),
    #     mu_e=Swept(GNConfig.Aggregation.DONTUSE, GNConfig.Aggregation.PRIMED),
    #     std_s=Swept(
    #         GNConfig.SAggregation.DONTUSE,
    #         GNConfig.SAggregation.PRIMED,
    #         GNConfig.SAggregation.SAMPLE,
    #     ),
    #     std_e=Swept(GNConfig.Aggregation.DONTUSE, GNConfig.Aggregation.PRIMED),
    # ),
    train_cfg=train_cfg,
    arch_cfg=acfg,
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(
            optim_momentum=(0.0),
            dec_momentum=Swept(False),
            bias_momentum=Swept(0.0),
            b2_technique=Swept("sq"),  # sq better
            b2_scale=Swept(1.0),
        ),
        bias_reset_value=-0.02,
        enc_directions=0,
        dec_directions=1,
        freq_balance=25,
    ),
)


class FreqBalanceSweep(SweepableConfig):
    run_cfg: RunConfig[Config] = runcfg
    # target_l0: int = Swept(2)
    # target_l0: int = Swept(2, 3, 5, 15, 25, 35, 50)
    target_l0: int | None = Swept(3, 6, 12, 25, 50)  # Swept(3, 6, 12, 25, 50)
    target_l1: int | float | None = Swept(
        1, 3, 9, 27, 81
    )  # Swept(1, 2, 4, 8, 32, 32 * 3)


cfg: FreqBalanceSweep = FreqBalanceSweep()

# cfg = cfg.random_sweep_configuration()


def run(cfg):
    tr = TrainingRunner(cfg.run_cfg, model_fn=sae)
    t = tr.trainer
    assert tr.normalizer.primed

    tr.resampler.assign_model(tr.trainable)
    tr.resampler.wholistic_freqbalance(
        model=tr.trainable,
        datasrc=tr.buf,
        target_l0=cfg.target_l0,
        target_l1=cfg.target_l1,
    )
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True, "rand" if quick_check else None)
