from saeco.data.data_config_definitions import (
    gpt_2_block,
)
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer.run_config import RunConfig
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.initializer import InitConfig

from saeco.architectures.vanilla import VanillaConfig, VanillaSAE


cfg = RunConfig[VanillaConfig](
    train_cfg=TrainConfig(
        data_cfg=gpt_2_block(),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=10_000,
            lr_warmup_length=2_000,
        ),
        #
        batch_size=4096,
        optim="Adam",
        # Wrapping the lr values in a Swept designates this
        # field to be swept across in the upcoming grid search
        lr=Swept(1e-3, 3e-4, 1e-4),
        betas=(0.9, 0.997),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=50,
        l0_target_adjustment_size=0.001,
        coeffs={
            "sparsity_loss": 1.1e-3,
            "L2_loss": 1,
        },
        #
        intermittent_metric_freq=1000,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=1,
    ),
    #
    init_cfg=InitConfig(d_data=768, dict_mult=8),
    arch_cfg=VanillaConfig(
        # And here we sweep across the pre_bias config option
        pre_bias=Swept(True, False),
    ),
)

arch = VanillaSAE(cfg)
sweep_manager = arch.get_sweep_manager()
sweep_manager.rand_run_no_agent()
# sweep_manager.initialize_sweep()

# # this will create 6 remote pods and begin running the grid search on them
# sweep_manager.run_manual_sweep_with_monitoring(new_pods=6)
