import os

import sys
from typing import TYPE_CHECKING

from saeco.architectures.vanilla import VanillaConfig, VanillaSAE
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import DataConfig, ModelConfig
from saeco.data.config.data_config_definitions import (
    gemma_2_2b_openwebtext,
    gemma_2_2b_openwebtext_bf16,
    gemma_2_2b_openwebtext_fp32,
    gemma_2_2b_openwebtext_test,
    gemma_2_2b_openwebtext_test_fp16,
    gemma_2_2b_openwebtext_test_fp32,
    gpt_2_block,
)
from saeco.data.config.generation_config import DataGenerationProcessConfig
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.data.config.split_config import SplitConfig
from saeco.initializer import InitConfig
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.sweep_expressions import SweepVar, Val, Var
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

PROJECT = "sae sweeps"


# var = SweepVar(1, 2, 3, name="var")
import saeco.core as cl


def acts_modifier(cache, acts):
    return acts + 1


cache = cl.Cache()
cache.register_write_callback("acts", acts_modifier)

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
