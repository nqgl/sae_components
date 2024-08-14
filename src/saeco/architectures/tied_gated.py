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
from saeco.trainer.RunConfig import RunConfig
from saeco.trainer.TrainConfig import TrainConfig
from saeco.trainer.runner import TrainingRunner
from saeco.data import DataConfig, ModelConfig, ActsDataConfig
from saeco.sweeps import Swept, do_sweep

from saeco.architectures.prolu.prolu import ProLUConfig, PProLU, thresh_from_bwd


class Config(SweepableConfig):
    pre_bias: bool = False
    clip_grad: float | None = Swept[float | None](1)
    # prolu_cfg: ProLUConfig = ProLUConfig(
    #     b_ste=1,
    #     m_ste=1,
    #     m_gg=1,
    # )
    # prolu_type_ste: float = Swept(0, 1)
    # det_prolu_type_ste: float = Swept(0, 1)
    max_shrink: float = Swept(0.3, 0.1)  ### Swept(0.3, 0.1, 0.03)
    gate_proportion: float = Swept(0.38, 0.05)  ### Swept(0.3, 0.1, 0.03)
    # DET_ENC_BIAS: bool = Swept(True, False)
    # SCALE_PRE: bool = False
    DET_ENC: bool = False  # Swept(True, False)  ### Swept(True, False)
    DET_DEC: bool = Swept(True, False)  ### Swept(True, False)
    # l1_on_full: float = 0
    l1_on_gate_only: bool = False
    enc_bias_init_value: float = 0  # Swept(0, -1)
    use_relu: bool = False
    prolu_gate: bool = False  # Swept(True, False)
    constrain_D: bool = Swept(True, False)

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


class DetProLu(PProLU):
    def forward(self, m):
        return self.prolu(m, self.bias.detach())


class ShrinkGate(cl.Module):
    def __init__(self, cfg, encoder, init):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        if self.cfg.prolu_gate:
            thresh = thresh_from_bwd(lambda x: 1)
            self.prolu = PProLU(
                ProLUConfig(b_ste=1, m_ste=1, m_gg=1), init._encoder.new_bias()
            )
            self.thresh = lambda m: thresh(self.prolu(m))
        else:
            self.thresh = thresh_from_bwd(lambda x: x > 0)

    def forward(self, x, shrinkgate, shrink, *, cache):
        gate = self.gate_eval(x)
        return gate

        # gate =

    def gate_eval(self, x):
        return self.thresh(self.encoder(x)).to(x.dtype)


class ShrinkGateSae(cl.Module):
    def __init__(
        self,
        init: Initializer,
        cfg: Config,
    ):
        super().__init__()
        self.cfg = cfg
        init._decoder.const_init_bias(0)
        init._encoder.const_init_bias(cfg.enc_bias_init_value)
        if cfg.constrain_D:
            init._decoder.add_wrapper(ft.NormFeatures)
            init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
        if self.cfg.prolu_gate:
            init._encoder._bias = False
        # self.prolu = PProLU(cfg.prolu_cfg, init.d_dict)
        # self.det_prolu = DetProLu(cfg.det_prolu_cfg, self.prolu.bias)
        # self.magnitude_encoder = nn.Linear(
        #     init.d_dict,
        # )
        self.actstuff = Seq(
            metrics=co.metrics.ActMetrics(),
            freqs=EMAFreqTracker(),
        )
        self.l1 = L1Penalty()
        # self.encode_pre = Seq(
        #     **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
        #     encoder_linear=init.encoder.resampled(),
        # )
        self.gate = ShrinkGate(cfg, init.encoder.resampled(), init=init)
        self.full_dec = init.decoder
        self.gate_dec = init._decoder.detached
        self.D_T = nn.Linear(
            init.d_data,
            init.d_dict,
            bias=False,
        )
        # if not cfg.constrain_D:
        #     self.D_T = co.LinEncoder(self.D_T)
        self.D_T.weight.data = self.full_dec.weight.transpose(-2, -1)

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
        if not self.training or (self.steps + 300) % 2500 < 30:
            if "skip" not in kwargs:
                v = self(x, cache=cache, skip=True)
                return v.detach() + 0 * v
            skipmul = 0

        pre_acts = cache(self).D_T(x)
        if self.cfg.use_relu:
            pre_acts = torch.relu(pre_acts)
        rand = 1 - skipmul * torch.rand_like(pre_acts, dtype=torch.float32)
        self.steps += 1

        shrinkmask = rand < self.cfg.gate_proportion
        shrink = 1 - self.cfg.max_shrink * rand / self.cfg.gate_proportion
        gate = cache(self).gate(x, shrinkmask, shrink)

        # torch.where(shrinkmask, gate * shrink, gate.detach())
        shrank_gate = torch.where(shrinkmask, gate * shrink, 0)
        full_gate = torch.where(~shrinkmask, gate.detach(), 0)
        if self.cfg.DET_ENC:
            pre_acts_gate = pre_acts.detach()
        else:
            pre_acts_gate = pre_acts
        acts_shrank = pre_acts_gate * shrank_gate
        acts_full = pre_acts * full_gate

        if self.cfg.l1_on_gate_only:
            cache(self).l1(shrank_gate)
        else:
            if self.cfg.DET_ENC:
                cache(self).l1(acts_shrank)
            else:
                cache(self).l1(pre_acts.detach() * shrank_gate)

        acts = acts_shrank + acts_full
        cache(self).actstuff(acts)

        if self.cfg.DET_DEC:
            out_full = cache(self).full_dec(acts_full)
            out_gate = cache(self).gate_dec(acts_shrank)
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
        "sparsity_loss": 3e-4,
        "L2_loss": 1,
    },
    lr=Swept(1e-4),
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0002,
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
        expected_biases=None,
        expected_encs=None,
    ),
)


class FreqBalanceSweep(SweepableConfig):
    run_cfg: RunConfig[Config] = runcfg
    # target_l0: int = Swept(2)
    # target_l0: int = Swept(2, 3, 5, 15, 25, 35, 50)
    target_l0: int | None = Swept(50)  # Swept(25)  # Swept(None, 6, 12)
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
