import torch

import torch.nn as nn

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

from saeco.architectures.prolu.prolu import ProLUConfig, PProLU


class Config(SweepableConfig):
    pre_bias: bool = False
    clip_grad: float | None = Swept[float | None](1)
    # prolu_cfg: ProLUConfig = ProLUConfig(
    #     b_ste=1,
    #     m_ste=1,
    #     m_gg=1,
    # )
    prolu_type_ste: float = Swept(0, 1)
    det_prolu_type_ste: float = Swept(0, 1)
    max_shrink: float = Swept(0.03, 0.01, 0)
    gate_proportion: float = 0.38
    DET_ENC: bool = False
    DET_ENC_BIAS: bool = False
    SCALE_PRE: bool = False
    DET_DEC: bool = False
    l1_on_full: float = 0
    det_prolu: bool = Swept(True, False)

    @property
    def prolu_cfg(self):
        return ProLUConfig(
            b_ste=self.prolu_type_ste,
            m_ste=self.prolu_type_ste,
            m_gg=1,
        )

    @property
    def det_prolu_cfg(self):
        return ProLUConfig(
            b_ste=self.det_prolu_type_ste,
            m_ste=self.det_prolu_type_ste,
            m_gg=1,
        )


def prorelu(m, b):
    return torch.relu(m) * (m + b > 0)


class DetProLU(PProLU):
    def forward(self, m):
        return self.prolu(m, self.bias.detach())


class ShrinkGateSae(cl.Module):
    def __init__(
        self,
        init: Initializer,
        cfg: Config,
    ):
        super().__init__()
        self.cfg = cfg
        init._decoder.const_init_bias()
        init._encoder.const_init_bias()
        init._decoder.add_wrapper(ft.NormFeatures)
        init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
        # init._encoder._bias = False
        self.prolu = PProLU(cfg.prolu_cfg, init.d_dict)
        if cfg.det_prolu:
            self.det_prolu = DetProLU(cfg.det_prolu_cfg, self.prolu.bias)
        else:
            self.det_prolu = PProLU(cfg.det_prolu_cfg, self.prolu.bias)

        self.actstuff = Seq(
            metrics=co.metrics.ActMetrics(),
            freqs=EMAFreqTracker(),
        )
        self.l1 = L1Penalty()
        self.encode_pre = Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            encoder_linear=init.encoder.resampled(),
        )
        self.full_dec = init.decoder
        self.gate_dec = init._decoder.detached
        self.gate_dec.use_bias = False
        self.steps = 0

    @torch.no_grad()
    def post_step_hook(self):
        return
        if self.steps > 700:
            return
        wl = [
            self.encode_pre.encoder_linear.features["weight"],
            self.full_dec.features["weight"],
        ]
        for w in wl:
            norm = w.features.norm(dim=1, keepdim=True)
            w.features[:] = torch.where(norm > 1, w.features / norm, w.features)

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        skipmul = 1
        if not self.training or (self.steps - 300) % 500 < 100:
            if "skip" not in kwargs:
                v = self(x, cache=cache, skip=True)
                return v.detach() + 0 * v
            skipmul = 0
        pre_acts = cache(self).encode_pre(x)
        rand = 1 - skipmul * torch.rand_like(pre_acts, dtype=torch.float32)
        self.steps += 1

        gatemask = rand < self.cfg.gate_proportion
        gate_scale = 1 - self.cfg.max_shrink * rand / self.cfg.gate_proportion
        if self.cfg.DET_ENC:
            pre_acts_gate = pre_acts.detach()
        elif self.cfg.DET_ENC_BIAS:
            pre_acts_gate = (
                pre_acts
                - self.encode_pre.encoder_linear.bias
                + self.encode_pre.encoder_linear.bias.detach()
            )
        else:
            pre_acts_gate = pre_acts
        if self.cfg.SCALE_PRE:
            acts_gate = cache(self).prolu(pre_acts_gate * gate_scale)
        else:
            acts_gate = cache(self).prolu(pre_acts_gate) * gate_scale
        acts_gate = torch.where(gatemask, acts_gate, 0)
        acts_full = cache(self).det_prolu(pre_acts)
        acts_full = torch.where(~gatemask, acts_full, 0)
        acts = acts_gate + acts_full
        cache(self).l1(
            acts_gate
            + acts_full
            * (
                max(
                    self.cfg.l1_on_full,
                    0,
                    # (max(1, 5000 / (1 + self.steps)) - 1) / 10000,
                )
            )
        )
        cache(self).actstuff(acts)
        if self.cfg.DET_DEC:
            out_full = cache(self).full_dec(acts_full)
            out_gate = cache(self).gate_dec(acts_gate)
            return out_full + out_gate
        return cache(self).full_dec(acts)


def sae(
    init: Initializer,
    cfg: Config,
):
    # init._encoder.add_wrapper(ReuseForward)
    model_full = ShrinkGateSae(init, cfg)
    if cfg.clip_grad:
        model_full = ClipGrad(model_full, cfg.clip_grad)
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
        run_length=50_000,
        resample_period=10_000,
        targeting_post_resample_hiatus=0,
        targeting_post_resample_cooldown=0.5,
        lr_resample_warmup_factor=0.3,
        targeting_delay=0,
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
    l0_target_adjustment_size=0.0003,
    batch_size=4096,
    use_lars=True,
    betas=(0.9, 0.999),
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
            optim_momentum=0.0,
            dec_momentum=False,
            bias_momentum=0.0,
            b2_technique="sq",  # sq better
            b2_scale=1.0,
        ),
        bias_reset_value=-0.02,
        enc_directions=0,
        dec_directions=1,
        freq_balance=None,
        expected_biases=2,
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
