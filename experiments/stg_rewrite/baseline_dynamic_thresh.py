from saeco.architectures.dynamic_thresh_prolu.model import (
    DynamicThreshConfig,
    DynamicThreshSAE,
    ThreshConfig,
)
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.config.data_config_definitions import (
    gpt_2_block,
)
from saeco.initializer import InitConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig


def s(x, *a):
    return x


cfg = RunConfig[DynamicThreshConfig](
    train_cfg=TrainConfig(
        data_cfg=gpt_2_block(7),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=50_000,
            lr_warmup_length=2_000,
        ),
        #
        batch_size=2048 * 2,
        optim="Adam",
        lr=1e-3,
        betas=(0.9, 0.99),
        #
        use_autocast=False,
        use_lars=True,
        #
        l0_target=50,
        l0_target_adjustment_size=0.001,
        coeffs={
            "sparsity_loss": 0.0e-3,
            "L2_loss": 1,
        },
        #
        intermittent_metric_freq=50000,
        use_averaged_model=False,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
    ),
    #
    init_cfg=InitConfig(d_data=768, dict_mult=8),
    arch_cfg=DynamicThreshConfig(
        thresh_cfg=ThreshConfig(
            decay_toward_mean=s(0.03),
            momentum=Swept(0.0),
            l0_diff_mult=30,
            lr=s(0.3),
            warmup_len=100,
            initial_value=0.5,
            stepclamp=s(1e-1),
            log_diff=False,
            # warmup_len=s(100),
        ),  # Swept(1.5, 2.0, 1.0, 0.7, 3.0),
        l1_end_scale=0.0,
        l1_decay_end=2_000,
        l1_decay_start=1_000,
    ),
)
arch = DynamicThreshSAE(cfg)
sweep_manager = arch.get_sweep_manager()
sweep_manager.rand_run_no_agent(project="nqgl/default-project")


# sweep_manager.initialize_sweep()
# sweep_manager.rand_run_no_agent()
# sweep_manager.run_manual_sweep_with_monitoring(new_pods=10)
