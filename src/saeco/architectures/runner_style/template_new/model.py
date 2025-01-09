import torch

import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
from saeco.components.model import Architecture
import saeco.core as cl

from saeco.initializer import Initializer
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)

from saeco.components.hooks.clipgrad import ClipGrad
from saeco.core import Seq

from saeco.misc import useif
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.Swept import Swept

from saeco.trainer.run_config import RunConfig
from .model import Config
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.initializer import InitConfig

from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


class Config(SweepableConfig):
    pre_bias: bool = False


def sae(
    init: Initializer,
    cfg: Config,
):
    model = Seq(
        encoder=Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            lin=init.encoder,
            nonlinearity=nn.ReLU(),
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        penalty=co.L1Penalty(),
        decoder=ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                init.decoder,
            ),
        ),
    )

    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return models, losses


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=sae)
    tr.trainer.train()


cfg = RunConfig[Config](
    train_cfg=TrainConfig(
        data_cfg=DataConfig(
            model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=True))
        ),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
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
    arch_cfg=Config(
        ...,
    ),
)

arch = Architecture(model_gen_fn=sae, base_cfg=cfg)
