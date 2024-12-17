from saeco.data.data_config_definitions import gpt_2_block
from saeco.trainer.run_config import RunConfig
from arch import GatedConfig, Gated
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.initializer import InitConfig

PROJECT = "sae sweeps"

cfg = RunConfig[GatedConfig](
    train_cfg=TrainConfig(
        data_cfg=gpt_2_block(layer=6),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=200,
            resample_period=12_500,
        ),
        #
        batch_size=4096,
        optim="Adam",
        lr=Swept(1e-3, 3e-4, 1e-4),
        betas=(0.9, 0.999),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=25,
        l0_target_adjustment_size=0.0002,
        coeffs={
            "sparsity_loss": 3e-3,
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
    ),
    #
    init_cfg=InitConfig(),
    arch_cfg=GatedConfig(),
)
g = Gated(cfg)
print()
cfg.is_concrete()
d = cfg.random_sweep_configuration()
g.instantiate(d.model_dump())
g.trainable
g.trainer.train()
print()
import torch

v = torch.randn(2, 768).cuda()
model = g.trainable

o = model(v)
o2 = model.decode(model.encode(v))
from pathlib import Path

p = Path.home() / "workspace/saved_models/Gated/201"

a2 = Gated.load_from_path(p, load_weights=True)
a2.trainable
