from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import DataConfig, ModelConfig
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

from .jumprelu_l1_model import Config

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
        batch_size=4096,
        optim="Adam",
        lr=Swept(1e-3),
        betas=(0.9, 0.999),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=25,
        l0_target_adjustment_size=0.0002,
        coeffs={
            "sparsity_loss": 3e-3,
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(), expected_biases=2
    ),
    #
    arch_cfg=Config(
        pre_bias=False,
        eps=Swept(3e-2),  ### Swept(1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3),
        thresh_initial_value=1,
        kernel="rect",  ### Swept("rect", "tri", "exp", "trap"),
        l1_begin_scale=Swept(0.3, 0.1, 0.03),
        l1_end_scale=Swept(0.0, 0.01),
        l1_end=Swept(20_000, 35_000, 49_000),
    ),
)
