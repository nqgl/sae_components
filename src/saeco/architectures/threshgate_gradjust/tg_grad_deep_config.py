from saeco.trainer.RunConfig import RunConfig
from .tg_grad_deep_model import DeepConfig
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.TrainConfig import TrainConfig
from saeco.initializer import InitConfig

PROJECT = "ezpod_test"

cfg = RunConfig[DeepConfig](
    train_cfg=TrainConfig(
        data_cfg=DataConfig(
            model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=True))
        ),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=Swept(50_000),
            resample_period=4000,
        ),
        #
        batch_size=4096,
        optim="ScheduleFree",
        lr=3e-3,
        betas=(0.9, 0.999),
        #
        use_autocast=False,
        use_lars=True,
        #
        l0_target=25,
        l0_target_adjustment_size=0.0003,
        l0_targeting_enabled=False,
        l0_targeter_type="pid",
        use_averaged_model=False,
        coeffs={
            "sparsity_loss": 3e-4,
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
        checkpoint_period=5000,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=None,
        expected_decs=None,
    ),
    #
    init_cfg=InitConfig(dict_mult=8),
    arch_cfg=DeepConfig(
        uniform_noise=True,
        noise_mult=1,
        # uniform_noise=Swept(True, False),
        # noise_mult=Swept(0.3, 1.0),
        decay_l1_to=0.03,
        mag_weights=False,
        leniency=0.35,
        leniency_targeting=True,
        deep_enc=False,
        deep_dec=1,
        # use_layernorm=Swept(True, False),
        # l1_max_only=Swept(True, False),
        use_layernorm=True,
        l1_max_only=False,
        penalize_after=False,
        resid=True,
        dec_mlp_expansion_factor=8,
        resample_dec=False,
        dec_mlp_nonlinearity="leakyrelu",
        norm_deep_dec=True,
    ),
)
