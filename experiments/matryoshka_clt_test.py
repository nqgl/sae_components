from saeco.architectures.matry_clt.matryoshka_clt import (
    MatryoshkaCLT,
    MatryoshkaCLTConfig,
)

from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.config.data_config_definitions import (
    gpt_2_block,
)
from saeco.initializer import InitConfig
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

cfg = RunConfig[MatryoshkaCLTConfig](
    train_cfg=TrainConfig(
        data_cfg=gpt_2_block([6, 7]),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=10_000,
            lr_warmup_length=2_000,
        ),
        #
        batch_size=16,
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
    init_cfg=InitConfig(d_data=768 * 2, dict_mult=8),
    arch_cfg=MatryoshkaCLTConfig(n_sites=2),
)

arch = MatryoshkaCLT(cfg)
sweep_manager = arch.get_sweep_manager()
sweep_manager.rand_run_no_agent(project="nqgl/default-project")

sweep_manager.initialize_sweep()

sweep_manager.run_manual_sweep_with_monitoring(new_pods=10)
