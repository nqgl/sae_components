import torch.nn as nn
import torch

from saeco.initializer import Initializer
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
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig
from saeco.trainer.runner import TrainingRunner
from saeco.data import DataConfig, ModelConfig, ActsDataConfig
from saeco.sweeps import Swept, do_sweep

from saeco.architectures.prolu.prolu import (
    ProLUConfig,
    PProLU,
    thresh_from_bwd,
    prolu_ste_from_thresh,
)


class Config(SweepableConfig):
    pre_bias: bool = False
    clip_grad: float | None = None
    # prolu_cfg: ProLUConfig = ProLUConfig(
    #     b_ste=1,
    #     m_ste=1,
    #     m_gg=1,
    # )
    anthropic_l1_mode: int = 2  # Swept(0, 1, 2)
    prolu_type_ste: float = 1
    from_thresh: None | str = None
    # Swept(
    #     None, "thresh", "0-1", "clamp-11", "clamp01", "1", "exp"
    # )
    bias_init_value: float = Swept(
        0, -0.1, -0.3, -0.7, -1, -1.25, -1.5, -1.75, -2, -2.25, -2.5
    )

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
    if not cfg.anthropic_l1_mode:
        init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
        init._decoder.add_wrapper(ft.NormFeatures)
    dec_mul_l1 = L1PenaltyScaledByDecoderNorm(det_dec_norms=cfg.anthropic_l1_mode == 1)
    init._decoder.const_init_bias()
    init._encoder.const_init_bias(cfg.bias_init_value)
    init._encoder._bias = False
    if cfg.from_thresh:
        if cfg.from_thresh == "thresh":
            bwd = lambda x: x > 0
        elif cfg.from_thresh == "0-1":
            bwd = lambda x: (x > 0) & (x < 1)
        elif cfg.from_thresh == "clamp-11":
            bwd = lambda x: x.clamp(-1, 1)
        elif cfg.from_thresh == "clamp01":
            bwd = lambda x: x.clamp(0, 1)
        elif cfg.from_thresh == "1":
            bwd = lambda x: 1
        elif cfg.from_thresh == "exp":
            bwd = lambda x: torch.exp(x)

        prolu = PProLU(
            prolu=prolu_ste_from_thresh(thresh_from_bwd(bwd)),
            d_bias=init._encoder.new_bias(),
        )
    else:
        prolu = PProLU(cfg.prolu_cfg, init._encoder.new_bias())
    model_full = Seq(
        **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
        encoder=Seq(
            linear=init.encoder.resampled(),
            prolu=prolu,
        ),
        freqs=EMAFreqTracker(),
        l1=dec_mul_l1 if cfg.anthropic_l1_mode else L1Penalty(),
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

from saeco.trainer.schedule_cfg import RunFloat

PROJECT = "sae sweeps"
quick_check = False
train_cfg = TrainConfig(
    optim="RAdam",  # Swept("RAdam", "NAdam", "Adam", "ScheduleFree", "ScheduleFreeAsNormal"),
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=not quick_check))
    ),
    raw_schedule_cfg=RunSchedulingConfig(
        run_length=70_000,
        resample_period=10_000,
        targeting_post_resample_hiatus=0,
        targeting_post_resample_cooldown=0.5,
        lr_resample_warmup_factor=0.3,
        lr_warmup_length=Swept[RunFloat](0.05),
        # resample_delay=0.69,
    ),
    l0_target=25,
    coeffs={
        "sparsity_loss": 4e-3,
        "L2_loss": 1,
    },
    lr=Swept(1e-4),
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0001,
    batch_size=4096,
    use_lars=True,
    betas=Swept[tuple[float, float]]((0.9, 0.999)),
)
acfg = Config(
    pre_bias=Swept[bool](False),
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
    target_l0: int | None = None  # Swept(None, 6, 12)
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
        datasrc=tr.data,
        target_l0=cfg.target_l0,
        target_l1=cfg.target_l1,
    )
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True, "rand" if quick_check else None)
