from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import (
    ActsDataConfig,
    DataConfig,
    DataGenerationProcessConfig,
    ModelConfig,
    SplitConfig,
)
from saeco.data.data_config_definitions import gemma_2_2b_openwebtext, gpt_2
from saeco.initializer import InitConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

from .model import Config, ThreshConfig

PROJECT = "sae sweeps"
cfg = RunConfig[Config](
    train_cfg=TrainConfig(
        data_cfg=gpt_2(),
        # data_cfg=gemma_2_2b_openwebtext,
        # data_cfg=DataConfig(
        #     model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=True)),
        #     # trainsplit=SplitConfig(start=0, end=40, tokens_from_split=50_000_000),
        # ),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=12_000,
            resample_period=9_000,
            lr_resample_warmup_length=0.1,
        ),
        #
        batch_size=4096,
        optim="Adam",
        lr=Swept(1e-3),
        betas=(0.9, 0.995),
        #
        use_autocast=True,
        use_lars=False,
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
    init_cfg=InitConfig(d_data=768, d_dict=8),
    #
    arch_cfg=Config(
        thresh_cfg=ThreshConfig(
            decay_toward_mean=10,  # Swept(0.1, 0.3, 1.0),
            initial_value=1,
        ),
        l1_end_scale=10_000,
        l1_decay_end=0,
        l1_decay_start=0,
    ),
)
