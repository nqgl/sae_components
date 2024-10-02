from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.data.data_config_definitions import gpt_2
from saeco.initializer import InitConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

from .stg_grad_deep_model import DeepConfig

PROJECT = "binarize"

cfg = RunConfig[DeepConfig](
    train_cfg=TrainConfig(
        data_cfg=gpt_2(),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=100_000,
            resample_period=4_000,
        ),
        #
        batch_size=4096,
        optim="Adam",
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=None,
        #
        use_autocast=False,
        use_lars=True,
        #
        l0_target=50,
        l0_target_adjustment_size=0.0003,
        l0_targeting_enabled=True,
        l0_targeter_type="pid",
        use_averaged_model=False,
        coeffs={
            "sparsity_loss": 1e-4,
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
        checkpoint_period=None,
        intermittent_metric_freq=5000,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(b2_scale=8),
        expected_biases=None,
        expected_decs=None,
    ),
    #
    init_cfg=InitConfig(dict_mult=8),
    arch_cfg=DeepConfig(
        uniform_noise=True,
        noise_mult=0.3,
        # uniform_noise=Swept(True, False),
        # noise_mult=Swept(0.3, 1.0),
        decay_l1_to=1,
        decay_l1_end=40_000,
        mag_weights=False,
        leniency=1,
        leniency_targeting=False,
        deep_enc=False,
        deep_dec=0,
        # use_layernorm=Swept(True, False),
        # l1_max_only=Swept(True, False),
        use_layernorm=False,
        l1_max_only=False,
        penalize_after=False,
        resid=True,
        dec_mlp_expansion_factor=4,
        resample_dec=True,
        dec_mlp_nonlinearity="prelu",
        norm_deep_dec=True,
        squeeze_channels=1,
        dropout=0.05,
        signed_mag=False,
    ),
)
