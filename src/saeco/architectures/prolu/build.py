import torch

import torch.nn as nn

from saeco.architectures.initialization.initializer import Initializer
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)

from saeco.components.hooks.clipgrad import ClipGrad
from saeco.core.reused_forward import ReuseForward
from saeco.core import Seq
import saeco.components.features.features as ft

import saeco.components as co
from saeco.misc import useif
from saeco.sweeps import SweepableConfig
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
import saeco.core as cl


from saeco.trainer import RunSchedulingConfig
from saeco.trainer.runner import TrainingRunner, TrainConfig, RunConfig
from saeco.data import DataConfig, ModelConfig, ActsDataConfig
from saeco.sweeps import Swept, do_sweep

from saeco.architectures.prolu.prolu import ProLUConfig, PProLU


class Config(SweepableConfig):
    pre_bias: bool = False
    clip_grad: float | None = Swept[float | None](None, 1)
    # prolu_cfg: ProLUConfig = ProLUConfig(
    #     b_ste=1,
    #     m_ste=1,
    #     m_gg=1,
    # )
    prolu_type_ste: float = Swept(-0.1, 0, 0.2, 0.5, 1)

    @property
    def prolu_cfg(self):
        return ProLUConfig(
            b_ste=self.prolu_type_ste,
            m_ste=self.prolu_type_ste,
            m_gg=1,
        )


def sae(
    init: Initializer,
    cfg: Config,
):
    # init._encoder.add_wrapper(ReuseForward)
    dec_mul_l1 = L1PenaltyScaledByDecoderNorm()
    init._decoder.zero_init_bias()
    init._encoder._bias = False
    model_full = Seq(
        **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
        encoder=Seq(
            linear=init.encoder.resampled(), prolu=PProLU(cfg.prolu_cfg, init.d_dict)
        ),
        freqs=EMAFreqTracker(),
        l1=dec_mul_l1,
        metrics=co.metrics.ActMetrics(),
        decoder=dec_mul_l1.set_decoder(init.decoder.resampled()),
    )
    if cfg.clip_grad:
        model_full = ClipGrad(model_full)
    models = [model_full]
    losses = dict(
        L2_loss=L2Loss(model_full),
        sparsity_loss=SparsityPenaltyLoss(model_full),
    )

    return models, losses


model_fn = sae

PROJECT = "sae sweeps"
quick_check = False
train_cfg = TrainConfig(
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=not quick_check))
    ),
    raw_schedule_cfg=RunSchedulingConfig(
        run_length=70_000,
        resample_period=10_000,
        targeting_post_resample_hiatus=0,
        targeting_post_resample_cooldown=0.5,
        lr_resample_warmup_factor=0.3,
        # resample_delay=0.69,
    ),
    l0_target=25,
    coeffs={
        "sparsity_loss": 4e-3,
        "L2_loss": 1,
    },
    lr=Swept(1e-4, 3e-5),
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0001,
    batch_size=4096,
    use_lars=True,
    betas=Swept[tuple[float, float]]((0.9, 0.999), (0.9, 0.99)),
)
acfg = Config(
    pre_bias=Swept[bool](True, False),
)
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)

runcfg = RunConfig[Config](
    train_cfg=train_cfg,
    arch_cfg=acfg,
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(
            optim_momentum=(0.0),
            dec_momentum=Swept(False),
            bias_momentum=Swept(0.0),
            b2_technique=Swept("sq"),  # sq better
            b2_scale=Swept(1.0),
        ),
        bias_reset_value=-0.02,
        enc_directions=0,
        dec_directions=1,
        freq_balance=25,
    ),
)


class FreqBalanceSweep(SweepableConfig):
    run_cfg: RunConfig[Config] = runcfg
    # target_l0: int = Swept(2)
    # target_l0: int = Swept(2, 3, 5, 15, 25, 35, 50)
    target_l0: int | None = Swept(25, None)  # Swept(None, 6, 12)
    target_l1: int | float | None = None  # Swept(None, 1, 4, 16, 64)


cfg: FreqBalanceSweep = FreqBalanceSweep()

# cfg = cfg.random_sweep_configuration()


def run(cfg):
    tr = TrainingRunner(cfg.run_cfg, model_fn=sae)
    t = tr.trainer
    assert tr.normalizer.primed

    tr.resampler.assign_model(tr.trainable)
    tr.resampler.wholistic_freqbalance(
        model=tr.trainable,
        datasrc=tr.buf,
        target_l0=cfg.target_l0,
        target_l1=cfg.target_l1,
    )
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True, "rand" if quick_check else None)
