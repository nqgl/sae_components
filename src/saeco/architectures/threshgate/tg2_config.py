from saeco.trainer.RunConfig import RunConfig
from .tg_model2 import Config, HardSoftConfig
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
            run_length=50_000,
            resample_period=90_000,
            lr_cooldown_length=0.3,
        ),
        #
        batch_size=4096,
        optim="Adam",
        lr=1e-3,
        betas=(0.9, 0.997),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=25,
        l0_target_adjustment_size=0.0003,
        l0_targeting_enabled=True,
        coeffs={
            "sparsity_loss": 1e-3,
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
        use_averaged_model=False,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=None,
        # expected_encs=1,
    ),
    #
    arch_cfg=Config(
        use_enc_bias=1,
        use_relu=Swept(True, False),
        hs_cfg=HardSoftConfig(
            p_soft=Swept(1.0, 0.75),
            noise_mag=None,
            eps=0.2,
            # mult_by_shrank=Swept(True, False),
            uniform_noise=Swept(True, False),
            noise_scale=Swept(0.02, 0.05, 0.1, 0.2),
            gate_backwards=Swept(True, False),
        ),
        hs_type="normrandsig",
        end_l1_penalty=0.001,
        detach=False,
    ),
)
