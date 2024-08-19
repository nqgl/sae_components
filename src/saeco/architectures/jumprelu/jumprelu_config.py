from saeco.trainer.RunConfig import RunConfig
from .jumprelu_model import Config
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.TrainConfig import TrainConfig

PROJECT = "sae sweeps"

cfg = RunConfig[Config](
    train_cfg=TrainConfig(
        data_cfg=DataConfig(
            model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=True))
        ),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000, resample_period=12_500, lr_cooldown_length=0.4
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
        l0_target=25,
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
    #
    arch_cfg=Config(
        pre_bias=False,
        eps=3e-3,
        thresh_initial_value=0.5,
        kernel="inv2",
        thresh_lr_mult=3,
        modified_jumprelu_grad=4,
        modified_thresh_grad=Swept(1, 2, 3),
        penalize_pre_acts=True,
        exp=False,
    ),
)
