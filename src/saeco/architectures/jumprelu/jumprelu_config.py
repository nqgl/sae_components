from saeco.trainer.run_config import RunConfig
from .jumprelu_model import Config
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig, SplitConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.initializer import InitConfig
from saeco.data.data_config_definitions import gemma_2_2b_openwebtext

PROJECT = "sae sweeps"
cfg = RunConfig[Config](
    train_cfg=TrainConfig(
        data_cfg=gemma_2_2b_openwebtext,
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=12_000,
            resample_period=12_500,
            # lr_cooldown_length=0.4
        ),
        #
        batch_size=4096 // 2,
        optim="Adam",
        lr=1e-3,
        betas=(0.9, 0.995),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=50,
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
    init_cfg=InitConfig(d_data=2048, d_dict=8),
    #
    arch_cfg=Config(
        pre_bias=False,
        eps=1e-2,
        thresh_initial_value=0.5,
        kernel="rect",
        thresh_lr_mult=1,
        modified_jumprelu_grad=0,
        modified_thresh_grad=0,
        penalize_pre_acts=True,
        exp=False,
    ),
)
