from saeco.trainer.RunConfig import RunConfig
from .model import AnthUpdateConfig
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.TrainConfig import TrainConfig
from saeco.initializer import InitConfig

PROJECT = "L0Targeting"

cfg = RunConfig[AnthUpdateConfig](
    train_cfg=TrainConfig(
        data_cfg=DataConfig(
            model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=True))
        ),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=12_500,
        ),
        #
        batch_size=4096,
        optim="Adam",
        lr=1e-3,
        betas=(0.9, 0.999),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=Swept(30.3),  # , 86.5, 308.8),
        l0_target_adjustment_size=Swept(0.0003, 0.0001, 0.001, 0.003),
        l0_targeter_type=Swept("pid", "basic", "gentle_basic"),
        l0_targeting_enabled=True,
        coeffs={
            "sparsity_loss": Swept(1e-2, 3e-2, 2e-4),
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
    ),
    #
    init_cfg=InitConfig(),
    arch_cfg=AnthUpdateConfig(),
)
