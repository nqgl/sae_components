from saeco.trainer.RunConfig import RunConfig
from .nf_hierarchical import HSAEConfig
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.TrainConfig import TrainConfig

PROJECT = "sae sweeps"


train_cfg = TrainConfig(
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=False)),
    ),
    raw_schedule_cfg=RunSchedulingConfig(
        run_length=10_000,
        resample_period=1_000,
        targeting_post_resample_hiatus=0.05,
        targeting_post_resample_cooldown=0.2,
        lr_resample_warmup_factor=0.1,
        # resample_delay=0.69,
    ),
    l0_target=22,
    coeffs={
        "sparsity_loss": Swept(1.5e-3),
        "L2_loss": 1,
    },
    lr=Swept(1e-3),
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0003,
    batch_size=4096,
    betas=Swept[tuple[float, float]]((0.9, 0.99)),
    use_lars=True,
    use_schedulefree=False,
    optim="ScheduleFree",
)

arch_cfg = HSAEConfig()

run_cfg = RunConfig[HSAEConfig](
    train_cfg=train_cfg,
    arch_cfg=arch_cfg,
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        bias_reset_value=0,
        enc_directions=0,
        dec_directions=1,
        freq_balance=2,
        dead_threshold=Swept(3e-5),
    ),
)

cfg = run_cfg
