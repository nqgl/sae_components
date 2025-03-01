from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer.run_config import RunConfig
from .model import SilodTopKConfig
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig

PROJECT = "sae sweeps"


train_cfg = TrainConfig(
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=True))
    ),
    raw_schedule_cfg=RunSchedulingConfig(
        run_length=50_000,
        resample_period=9_000,
        targeting_post_resample_hiatus=0.05,
        targeting_post_resample_step_size_warmup=0.2,
        lr_resample_warmup_factor=0.3,
        # resample_delay=0.69,
    ),
    l0_target=25,
    coeffs={
        "sparsity_loss": 3e-3,
        "L2_loss": 1,
        "L2_aux_loss": 1 / 32,
    },
    optim=Swept("ScheduleFree"),
    lr=Swept(3e-3, 1e-3, 3e-4),
    use_autocast=True,
    use_lars=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0002,
    batch_size=4096,
    betas=(0.9, 0.999),
)

arch_cfg = SilodTopKConfig(
    pre_bias=False,
    skew=Swept[float](1, 2, 3),
    num_silos=Swept(1, 2, 4, 8, 16, 25),
)

run_cfg = RunConfig[SilodTopKConfig](
    train_cfg=train_cfg,
    arch_cfg=arch_cfg,
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        dead_threshold=1.2e-5,
    ),
)

cfg = run_cfg
