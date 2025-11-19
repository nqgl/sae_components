from saeco.data.data_config_definitions import gemma_2_2b_openwebtext_bf16

from saeco.architectures.dynamic_thresh_prolu.model import (
    DynamicThreshConfig,
    DynamicThreshSAE,
    ThreshConfig,
)
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.initializer import InitConfig
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

cfg = RunConfig[DynamicThreshConfig](
    train_cfg=TrainConfig(
        data_cfg=gemma_2_2b_openwebtext_bf16(),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=10_000,
            lr_warmup_length=2_000,
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
        l0_target=50,
        l0_target_adjustment_size=0.001,
        coeffs={
            "sparsity_loss": 0,
            "L2_loss": 1,
        },
        #
        intermittent_metric_freq=5000,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
    ),
    #
    init_cfg=InitConfig(d_data=2304, dict_mult=8),
    arch_cfg=DynamicThreshConfig(
        thresh_cfg=ThreshConfig(
            decay_toward_mean=0.1,
            momentum=0.0,
            l0_diff_mult=1,
            lr=0.03,
            warmup_len=0,
            initial_value=1,
        ),  # Swept(1.5, 2.0, 1.0, 0.7, 3.0),
        l1_end_scale=0,  # Swept(0.0, 0.01),
    ),
)

arch = DynamicThreshSAE(cfg)
sweep_manager = arch.get_sweep_manager()
sweep_manager.rand_run_no_agent(project="nqgl/default-project")

# sweep_manager.initialize_sweep()

# sweep_manager.run_manual_sweep_with_monitoring(new_pods=10)
