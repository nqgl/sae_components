from .model import TopKConfig
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
    optim=Swept("ScheduleFree", "ScheduleFreeAsNormal"),
    lr=Swept(3e-3),
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0002,
    batch_size=4096,
    betas=(0.9, 0.999),
    use_lars=Swept(True, False),
)

arch_cfg = TopKConfig(
    pre_bias=Swept[bool](False),
)

run_cfg = RunConfig[TopKConfig](
    train_cfg=train_cfg,
    arch_cfg=arch_cfg,
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
    ),
)

cfg = run_cfg
