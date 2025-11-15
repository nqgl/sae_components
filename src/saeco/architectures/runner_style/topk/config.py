from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer.run_config import RunConfig
from .model import TopKConfig
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import DataConfig, ModelConfig
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
        resample_period=Swept(100_000, 3_000),
        targeting_post_resample_hiatus=0.05,
        targeting_post_resample_step_size_warmup=0.2,
        lr_resample_warmup_factor=0.3,
        # resample_delay=0.69,
    ),
    l0_target=25,
    coeffs={"sparsity_loss": 3e-3, "L2_loss": 1, "L2_aux_loss": Swept(1 / 32, 1.0)},
    optim="Adam",
    lr=Swept(1e-3, 3e-4),
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0002,
    batch_size=4096,
    betas=(0.9, 0.999),
    use_lars=True,
)

arch_cfg = TopKConfig(
    pre_bias=False, aux_top_k=Swept(0, 32, 512), dead_threshold=Swept(3e-5, 1e-5, 3e-6)
)

run_cfg = RunConfig[TopKConfig](
    train_cfg=train_cfg,
    arch_cfg=arch_cfg,
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(), dead_threshold=1.1e-5
    ),
)

cfg = run_cfg
