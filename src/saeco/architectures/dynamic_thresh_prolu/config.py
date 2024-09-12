from saeco.trainer.run_config import RunConfig
from .model import Config, ThreshConfig
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.data.data_config_definitions import gemma_2_2b_openwebtext
from saeco.initializer import InitConfig

PROJECT = "sae sweeps"
batch_exp = 2
cfg = RunConfig[Config](
    train_cfg=TrainConfig(
        data_cfg=gemma_2_2b_openwebtext,
        # DataConfig(model_cfg=ModelConfig(acts_cfg=ActsDataConfig())),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=6_000 * batch_exp,
            resample_period=9_000,
            lr_resample_warmup_length=0.1,
        ),
        #
        batch_size=4096 // batch_exp,
        optim="Adam",
        lr=Swept(1e-3),
        betas=(0.9, 0.995),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=50,
        l0_targeting_enabled=False,
        l0_target_adjustment_size=0.0002,
        coeffs={
            "sparsity_loss": Swept(0.0),
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
        bias_reset_value=0,
    ),
    init_cfg=InitConfig(d_data=2048, d_dict=8),
    #
    arch_cfg=Config(
        thresh_cfg=ThreshConfig(
            decay_toward_mean=Swept(0.1, 0.3, 1.0),
            initial_value=1,
        ),
        l1_end_scale=0,
        l1_decay_end=3_000 * batch_exp,
        l1_decay_start=0,
    ),
)
