from saeco.data.data_config_definitions import (
    gpt_2_block,
    gemma_2_2b_openwebtext_test,
    gemma_2_2b_openwebtext,
    gemma_2_2b_openwebtext_test_fp16,
    gemma_2_2b_openwebtext_test_fp32,
    gemma_2_2b_openwebtext_fp32,
    gemma_2_2b_openwebtext_bf16,
)
from saeco.data.generation_config import DataGenerationProcessConfig
from saeco.data.split_config import SplitConfig
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config.sweep_expressions import Val
from saeco.trainer.run_config import RunConfig
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig, Swept, SweepVar
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.initializer import InitConfig

import sys
import os
from typing import TYPE_CHECKING

from saeco.architectures.vanilla import Config, VanillaSAE

PROJECT = "sae sweeps"

# var = SweepVar[int](1, 2, name="var")
batch_size_mult_var = SweepVar[int](1, 2, name="batch_size_mult")

val50k_int = Val[int](value=50_000)
val50k_int.generic_type
val50k = Val(value=50_000)
val50k.generic_type
print()
raw_schedule_cfg = RunSchedulingConfig(
    run_length=Val[int](value=50_000) // batch_size_mult_var,
    resample_period=Val[int](value=8_000) // batch_size_mult_var,
    lr_cooldown_length=0.3,
    lr_warmup_length=2000,
)
dictval = Val[dict[int, int]](value={1: 22, 2: 33, 3: 55})

cfg = RunConfig[Config](
    train_cfg=TrainConfig(
        data_cfg=gpt_2_block(),
        raw_schedule_cfg=raw_schedule_cfg,
        #
        batch_size=batch_size_mult_var * 4096,
        optim="Adam",
        lr=Swept(1e-3, 2e-3),
        betas=(0.9, 0.997),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=50,
        l0_target_adjustment_size=0.001,
        coeffs={
            "sparsity_loss": 1e-3,
            "L2_loss": 1.0,
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
    init_cfg=InitConfig(d_data=768, dict_mult=Swept(*[2**i for i in range(2, 4)])),
    arch_cfg=Config(),
)
# from transformers import Gemma2ForCausalLM

# import saeco.data.model_cfg as mc

# # mc.MODEL_FN_CALLABLE_OVERRIDE = Gemma2ForCausalLM.from_pretrained

from saeco.sweeps.sweepable_config.SweptNode import check_config_combinations

check_config_combinations(cfg)
g = VanillaSAE(cfg)
sweep_manager = g.get_sweep_manager()
print(sweep_manager.initialize_sweep(project="project2", custom_sweep=True))
# sweep_manager.rand_run_no_agent()
# sweep_manager.local_sweep()
# sweep_manager.get_worker_run_command()
sweep_manager.get_worker_run_commands_for_manual_sweep()
sweep_manager.run_manual_sweep_with_monitoring(8, purge_after=True, setup_min=8)
sweep_manager.run_sweep_on_pods_with_monitoring(8, purge_after=True, setup_min=8)


sweep_manager.initialize_sweep()
print()
cfg.is_concrete()
d = cfg.random_sweep_configuration()
g.instantiate(d.model_dump())
import nnsight
import torch
import einops

g.trainer.train()


def normal_runner(model: nnsight.LanguageModel, cfg: ActsDataConfig, skip_first=False):
    def nrunner(tokens):
        return model.trace(tokens, trace=False)

    return nrunner


def to_losses(model_callable):
    def runner(tokens: torch.Tensor):
        out = model_callable(tokens)
        l = einops.rearrange(
            out.logits[:, :-1], "batch seq vocab -> (batch seq) vocab"
        ).cuda()
        t = einops.rearrange(tokens[:, 1:], "batch seq -> (batch seq)").cuda()
        loss = torch.nn.functional.cross_entropy(l, t)

        return loss

    return runner


model = cfg.train_cfg.data_cfg.model_cfg.model
normal = to_losses(
    normal_runner(
        model,
        cfg.train_cfg.data_cfg.model_cfg.acts_cfg,
        skip_first=True,
    )
)
with torch.inference_mode():
    input_text = "Write me a poem about Machine Learning."
    input_ids = model.tokenizer(input_text, return_tensors="pt")
    input_ids["input_ids"].dtype
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        normal(input_ids["input_ids"].cuda())

print("done")
