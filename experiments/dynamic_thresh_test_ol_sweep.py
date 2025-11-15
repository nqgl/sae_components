from saeco.architectures.dynamic_thresh_prolu.model import (
    DynamicThreshConfig,
    DynamicThreshSAE,
    ThreshConfig,
)
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.data_config_definitions import gemma_2_2b_openwebtext_bf16, gpt_2_block
from saeco.initializer import InitConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import EarlyStoppingBounds, TrainConfig


def s(x, *a):
    return Swept(x / 10, x / 3, x, x * 3, x * 10, *a)


cfg = RunConfig[DynamicThreshConfig](
    train_cfg=TrainConfig(
        save_on_complete=False,
        data_cfg=gpt_2_block([6, 7]),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=4_500 * 4,
            resample_period=10_000,
            lr_warmup_length=600,
        ),
        #
        batch_size=4096 // 4,
        optim=Swept("AdamW", "NAdam", "ScheduleFree", "AmsgradW", "Adamax"),
        lr=s(1e-3 / 1.5),
        weight_decay=Swept[float | None](None, 0.3, 0.1, 0.03, 0.003),
        betas=Swept((0.9, 0.99), (0.9, 0.9), (0.9, 0.95)),
        #
        use_autocast=False,
        use_lars=Swept(True, False),
        #
        l0_target=50,
        l0_target_adjustment_size=Swept(0.001, 0.0001),
        coeffs={
            "sparsity_loss": Swept(0, 0, 3e-5),
            "L2_loss": 1,
        },
        #
        intermittent_metric_freq=5000,
        early_stopping_bounds=EarlyStoppingBounds(
            min_values={
                "L0": {
                    10: 1.0,
                    100: 1.0,
                    1000: 5.0,
                    1500: 10.0,
                    2000: 15.0,
                    2500: 15.0,
                    3000: 20.0,
                    4000: 30.0,
                }
            },
            max_values={
                "L2_loss": {
                    100: 3.0,
                    500: 1.0,
                    1000: 0.35,
                    2000: 0.3,
                }
                | {i: 3.0 for i in range(0, 4000, 100)},
                "L0": {
                    1000: 500.0,
                    4000: 70.0,
                }
                | {1000 + 100 * i: 120.01 + 40 * 8 - 8 * i for i in range(40)},
            },
        ),
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
    ),
    #
    init_cfg=InitConfig(d_data=1024 + 512, dict_mult=128),
    arch_cfg=DynamicThreshConfig(
        thresh_cfg=ThreshConfig(
            decay_toward_mean=s(0.1, 0.0),
            momentum=Swept(0.0, 0.1),
            l0_diff_mult=s(10),
            lr=s(0.3),
            warmup_len=100,
            initial_value=s(1),
            stepclamp=s(1e-1),
            log_diff=Swept(True, False),
            # warmup_len=s(100),
        ),  # Swept(1.5, 2.0, 1.0, 0.7, 3.0),
        l1_end_scale=Swept(0.0, 0.01),
        l1_decay_end=2_000,
        l1_decay_start=1_000,
    ),
)

arch = DynamicThreshSAE(cfg)
sweep_manager = arch.get_sweep_manager()
sweep_manager.rand_run_no_agent(project="nqgl/default-project")

# sweep_manager.initialize_sweep()

# sweep_manager.run_manual_sweep_with_monitoring(new_pods=10)
