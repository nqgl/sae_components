from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer.run_config import RunConfig
from saeco.initializer import InitConfig
from .dg_gate import Config
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig

PROJECT = "sae sweeps"

cfg = RunConfig[Config](
    train_cfg=TrainConfig(
        data_cfg=DataConfig(
            model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=True))
        ),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=12_500,
        ),
        #
        batch_size=2048,
        optim="ScheduleFree",
        lr=3e-3,  # Swept(1e-3, 3e-3, 1e-2),  # Swept(1e-3, 3e-4, 1e-4),
        betas=(0.9, 0.999),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=25,
        l0_target_adjustment_size=0.001,
        coeffs={
            "sparsity_loss": 3e-3 / 32,
            "L2_loss": 1,
            "L2_aux_loss": 1 / 32,
        },
        #
        wandb_cfg=dict(project=PROJECT),
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
    ),
    #
    arch_cfg=Config(
        relu_first_acts=False,
        gelu_mid=False,
    ),
    init_cfg=InitConfig(dict_mult=16),
)
