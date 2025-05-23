import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
from saeco.components import EMAFreqTracker, L1Penalty, L2Loss, SparsityPenaltyLoss
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward

from saeco.initializer import Initializer
from saeco.initializer.initializer_config import InitConfig
from saeco.misc import useif
from saeco.sweeps import SweepableConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.schedule_cfg import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig


from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.data.data_config_definitions import gpt_2_block
from saeco.architectures.vanilla.vanilla_model import VanillaSAE, VanillaConfig

PROJECT = "sae sweeps"
train_cfg = TrainConfig(
    data_cfg=gpt_2_block(),
    l0_target=25,
    coeffs={
        "sparsity_loss": 4e-3,
        "L2_loss": 1,
    },
    lr=1e-3,
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0003,
    batch_size=4096,
    use_lars=True,
    betas=(0.9, 0.999),
    raw_schedule_cfg=RunSchedulingConfig(
        run_length=1000,
        lr_cooldown_length=0.2,
        lr_resample_warmup_length=0.3,
        lr_resample_warmup_factor=0.2,
        # targeting_resample_cooldown_period_override=(2_000),
        targeting_post_resample_step_size_warmup=0.3,
        targeting_post_resample_hiatus=0.4,
        resample_period=(4_000),
        resampling_finished_phase=0.3,
        resample_delay=(5_000),
    ),
)

cfg = RunConfig[VanillaConfig](
    train_cfg=train_cfg,
    arch_cfg=VanillaConfig(),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
    ),
    init_cfg=InitConfig(dict_mult=16),
)

arch = VanillaSAE(cfg)

data = arch.run_cfg.train_cfg.data_cfg._get_databuffer(num_workers=16)
model = arch.trainable
import torch
import tqdm

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
# for i in tqdm.trange(800):
#     x = next(data).cuda()
#     y = model(x)
#     l = (y - x).pow(2).mean()
#     l.backward()
#     optim.step()
#     optim.zero_grad()
# with torch.no_grad():
#     q = 0
#     print("disk + sum task")

#     for i in tqdm.trange(1500):
#         x = next(data)
#         q += x
data = arch.data

for _ in range(3):
    for i in tqdm.trange(100):
        x = next(data).cuda()
        y = model(x)
        l = (y - x).pow(2).mean()
        l.backward()
        optim.step()
        optim.zero_grad()
with torch.no_grad():
    q = 0
    print("disk + sum task")
    for _ in range(3):
        for i in tqdm.trange(101):
            x = next(data)
            q += x

data = arch.trainer.get_databuffer(num_workers=16, queue_size=128)


for _ in range(8):
    for i in tqdm.trange(100):
        x = next(data).cuda()
        y = model(x)
        l = (y - x).pow(2).mean()
        l.backward()
        optim.step()
        optim.zero_grad()
with torch.no_grad():
    q = 0
    print("disk + sum task")
    for _ in range(15):
        for i in tqdm.trange(101):
            x = next(data)
            q += x

# data = arch.data

# print("done")
# for _ in range(8):
#     for i in tqdm.trange(99):
#         x = next(data).cuda()
#         y = model(x)
#         l = (y - x).pow(2).mean()
#         l.backward()
#         optim.step()
#         optim.zero_grad()
# with torch.no_grad():
#     q = 0
#     print("disk + sum task")
#     for _ in range(15):
#         for i in tqdm.trange(88):
#             x = next(data)
#             q += x

# for _ in range(8):
#     for i in tqdm.trange(99):
#         x = next(data).cuda()
#         y = model(x)
#         l = (y - x).pow(2).mean()
#         l.backward()
#         optim.step()
#         optim.zero_grad()
# with torch.no_grad():
#     q = 0
#     print("disk + sum task")
#     for _ in range(15):
#         for i in tqdm.trange(88):
#             x = next(data)
#             q += x
