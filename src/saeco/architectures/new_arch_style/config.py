from saeco.data.config.data_config_definitions import (
    gpt_2_block,
    gemma_2_2b_openwebtext_test,
    gemma_2_2b_openwebtext,
    gemma_2_2b_openwebtext_test_fp16,
    gemma_2_2b_openwebtext_test_fp32,
    gemma_2_2b_openwebtext_fp32,
    gemma_2_2b_openwebtext_bf16,
)
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer.run_config import RunConfig
from saeco.architectures.new_arch_style.arch import GatedConfig, Gated
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.initializer import InitConfig

PROJECT = "sae sweeps"

cfg = RunConfig[GatedConfig](
    train_cfg=TrainConfig(
        data_cfg=gemma_2_2b_openwebtext_bf16(17),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=120_100,
            resample_period=12_500,
            lr_cooldown_length=0.3,
            lr_warmup_length=500,
        ),
        #
        batch_size=4096 * 2,
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
        expected_biases=2,
    ),
    #
    init_cfg=InitConfig(d_data=2304, dict_mult=8),
    arch_cfg=GatedConfig(),
)
# from transformers import Gemma2ForCausalLM

# import saeco.data.model_cfg as mc

# # mc.MODEL_FN_CALLABLE_OVERRIDE = Gemma2ForCausalLM.from_pretrained
g = Gated(cfg)
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
