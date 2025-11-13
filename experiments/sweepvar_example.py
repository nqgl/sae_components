from saeco.data.config.data_config_definitions import (
    gpt_2_block,
)
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.sweeps.sweepable_config.sweep_expressions import (
    SweepVar,
    Var,
)
from saeco.trainer.run_config import RunConfig
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.sweeps.sweepable_config.sweep_expressions import Val
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.initializer import InitConfig

from saeco.architectures.vanilla import VanillaConfig, VanillaSAE
import saeco.core as cl


batch_size_mult_var = SweepVar(1, 2, 3, name="batch_size_mult")
cfg = RunConfig[VanillaConfig](
    train_cfg=TrainConfig(
        data_cfg=gpt_2_block(),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=Val(50_000) // batch_size_mult_var,
            resample_period=Val(8_000) // batch_size_mult_var,
            lr_cooldown_length=0.5,
            lr_warmup_length=500,
        ),
        #
        batch_size=batch_size_mult_var * 4096,
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
            "sparsity_loss": 1.1e-3,
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
        intermittent_metric_freq=1000,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=1,
    ),
    #
    init_cfg=InitConfig(d_data=768, dict_mult=8),
    arch_cfg=VanillaConfig(pre_bias=Swept(True, False)),
)
g = VanillaSAE(cfg)
sweep_manager = g.get_sweep_manager()
sweep_manager.initialize_sweep()
sweep_manager.run_sweep_on_pods_with_monitoring(
    2, purge_after=False, keep_after=True, challenge_file=None
)


g = VanillaSAE(cfg)
g.run_training()
