from saeco.architectures.anth_update.model import Config
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.runner import RunConfig, TrainConfig

PROJECT = "sae sweeps"


train_cfg = TrainConfig(
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=True))
    ),
    raw_schedule_cfg=RunSchedulingConfig(
        run_length=50_000,
        resample_period=250_000,
        targeting_post_resample_hiatus=0.05,
        targeting_post_resample_cooldown=0.2,
        lr_resample_warmup_factor=0.3,
        # resample_delay=0.69,
    ),
    l0_target=25,
    coeffs={
        "sparsity_loss": 3e-3,
        "L2_loss": 1,
    },
    lr=Swept(1e-2, 3e-3, 1e-3, 3e-4, 1e-4),
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0001,
    batch_size=4096,
    betas=Swept[tuple[float, float]](
        (0.9, 0.999), (0.93, 0.999), (0.95, 0.999), (0.97, 0.999)
    ),
    use_lars=True,
    use_schedulefree=False,
)

arch_cfg = Config(
    pre_bias=Swept[bool](False),
)

run_cfg = RunConfig[Config](
    train_cfg=train_cfg,
    arch_cfg=arch_cfg,
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
        freq_balance=None,
    ),
)

cfg = run_cfg
