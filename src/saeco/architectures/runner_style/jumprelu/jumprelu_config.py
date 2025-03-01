from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig, SplitConfig
from saeco.data.data_config_definitions import gemma_2_2b_openwebtext, gpt_2_block
from saeco.initializer import InitConfig
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

from .jumprelu_model import Config

PROJECT = "sae sweeps"
cfg = RunConfig[Config](
    train_cfg=TrainConfig(
        data_cfg=gpt_2_block(layer=6),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=8_000,
            resample_period=9_000,
            # lr_cooldown_length=0.4
        ),
        #
        batch_size=4096,
        optim="Adam",
        lr=Swept(3e-4, 1e-3, 3e-4),
        betas=(0.9, 0.997),
        #
        use_autocast=False,
        use_lars=False,
        #
        l0_target=100,
        l0_target_adjustment_size=0.0003,
        coeffs={
            "sparsity_loss": 7e-4,
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(), expected_biases=2
    ),
    init_cfg=InitConfig(d_data=768, dict_mult=64),
    #
    arch_cfg=Config(
        pre_bias=False,
        eps=Swept(1e-2, 1e-3, 1e-4, 3e-4, 3e-3, 3e-2, 1e-1),
        thresh_initial_value=0.5,
        kernel=Swept("rect", "gauss", "tri"),
        thresh_lr_mult=1,
        # modified_jumprelu_grad=Swept(2, 3, 4, 5),
        # modified_thresh_grad=Swept(2, 3, 4, 5),
        penalize_pre_acts=True,
        exp=False,
    ),
)
