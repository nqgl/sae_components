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
from saeco.sweeps.sweepable_config.sweep_expressions import SweepVar, Val
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

PROJECT = "sae sweeps"


import saeco.core as cl

from saeco.mlog import mlog

mlog.init()


def acts_modifier(cache, acts):
    return acts + 1


cache = cl.Cache()
cache.register_write_callback("acts", acts_modifier)


def gpt_2(block_postfix):
    return DataConfig(
        dataset="alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
        model_cfg=ModelConfig(
            acts_cfg=ActsDataConfig(
                excl_first=True,
                sites=(["transformer.h.3.input", "transformer.h.4.input"]),
                d_data=768,
                autocast_dtype_str="bfloat16",
                force_cast_dtype_str="bfloat16",
                storage_dtype_str="bfloat16",
            ),
            model_name="gpt2",
        ),
        trainsplit=SplitConfig(start=0, end=50, tokens_from_split=2_000_000),
        generation_config=DataGenerationProcessConfig(
            # tokens_per_pile=2**25,
            acts_per_pile=2**17,
            meta_batch_size=2**19,
            llm_batch_size=2**16,
        ),
        seq_len=256,
    )


def gpt_2_block(layer: int | list[int] | tuple[int], io="input"):
    if isinstance(layer, list | tuple):
        return gpt_2([f"{l}.{io}" for l in layer])
    return gpt_2(f"{layer}.{io}")


cfg = RunConfig[VanillaConfig](
    train_cfg=TrainConfig(
        data_cfg=gpt_2_block(3),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=200,
            resample_period=8_000,
            lr_cooldown_length=0.5,
            lr_warmup_length=500,
        ),
        #
        batch_size=2 * 4096,
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
        input_sites=["transformer.h.3.input"],
        target_sites=["transformer.h.4.input"],
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=1,
    ),
    #
    init_cfg=InitConfig(d_data=768, dict_mult=8),
    arch_cfg=VanillaConfig(pre_bias=True),
)

g = VanillaSAE(cfg)
g.run_training()
