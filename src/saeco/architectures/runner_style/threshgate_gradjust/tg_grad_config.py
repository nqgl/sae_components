from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.initializer import InitConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

from .tg_grad_model import Config

PROJECT = "sae sweeps"

cfg = RunConfig[Config](
    train_cfg=TrainConfig(
        data_cfg=DataConfig(
            model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=True))
        ),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=4000,
        ),
        #
        batch_size=4096,
        optim="Adam",
        lr=Swept(1e-3, 3e-4),
        betas=(0.9, 0.999),
        #
        use_autocast=False,
        use_lars=True,
        #
        l0_target=25,
        l0_target_adjustment_size=0.0003,
        l0_targeting_enabled=Swept(True, False),
        use_averaged_model=False,
        coeffs={
            "sparsity_loss": 3e-4,
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(), expected_biases=None
    ),
    #
    init_cfg=InitConfig(dict_mult=8),
    arch_cfg=Config(
        uniform_noise=False,
        noise_mult=0.3,
        exp_mag=False,
        leniency_targeting=Swept(True, False),
        decay_l1=Swept(True, False),
        mag_weights=Swept(True, False),
    ),
)
