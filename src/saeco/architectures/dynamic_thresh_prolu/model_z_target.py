import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
import torch
from saeco.components.metrics.metrics import PreActMetrics

import torch.nn as nn
from pydantic import Field
from saeco.components import (
    EMAFreqTracker,
    FreqTracker,
    L1Penalty,
    L2Loss,
    LinearDecayL1Penalty,
    SparsityPenaltyLoss,
)

from saeco.components.hooks.clipgrad import ClipGrad
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.core import Seq
from saeco.initializer import Initializer

from saeco.misc import useif
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.architecture import (
    Architecture,
    arch_prop,
    loss_prop,
    model_prop,
    aux_model_prop,
)

# torch.backends.cudnn.benchmark = True


class ThreshConfig(SweepableConfig):
    # eps: float = Swept(1e1)
    logeps: float = 3e-7
    lr: float = 0.03  # Swept(0.1, 0.01)
    # signstep: float = 0
    warmup_len: float = 5_000
    momentum: float = 0.9
    stepclamp: None | float = 0.01
    decay_toward_mean: float = 0.1

    # Swept(1e-2, 3e-2, 1e-3, 3e-3)  # Swept(
    # 1e-2, 1e-3, 1e-4
    # )  # Swept(0.003, 0.001, 0.0003, 0.0001)
    log_diff: bool = True
    l0_diff_mult: float = 30
    initial_value: float = 0.5
    zipf: bool = True
    zipf_r: float = 0.5
    zipf_i: int = 10  # Swept(1, 10, 20)
    # l0_targ_lin: bool = Swept(True, False)

    # min_freq_ratio: float | None = Swept(3, 10)
    # max_freq_ratio: float | None = Swept(3, 10)
    # @property
    # def min_freq_ratio(self):
    #     return self.freq_ratios

    # @property
    # def max_freq_ratio(self):
    #     return self.freq_ratios


class DynamicZipfThreshConfig(SweepableConfig):
    thresh_cfg: ThreshConfig = Field(default_factory=ThreshConfig)
    pre_bias: bool = False
    l1_decay_start: int = 5_000
    l1_decay_end: int = 25_000
    l1_end_scale: float = 0.0


import math


def log(x):
    if isinstance(x, torch.Tensor):
        return x.log()
    else:
        return math.log(x)


class Thresholder(cl.Module):
    thresh_values: torch.Tensor
    thresh_value_momentum: torch.Tensor

    def __init__(self, init: Initializer, cfg: ThreshConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "thresh_values", torch.zeros(init.d_dict) + cfg.initial_value
        )
        self.register_buffer("thresh_value_momentum", torch.zeros(init.d_dict))
        self.nonlinearity = nn.ReLU()
        self.target = init.l0_target
        self.freqs: co.FreqTracker = None
        self.freq_target_constant = init.l0_target / init.d_dict
        self.freq_target = self.freq_target_constant
        self.t = 0
        # self.prev_cache = None
        self.d_dict = init.d_dict

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        x = cache(self).nonlinearity(x)
        # self.prev_cache = cache
        return torch.where(
            x > self.thresh_values.unsqueeze(0),
            x,
            0,
        )

    @torch.no_grad()
    def zipf(self, i: int = 1, r: float = 1.0):
        z = torch.arange(i, self.d_dict + i).float().pow(r).reciprocal()
        z[self.freqs.freqs.argsort(descending=True)] = z.clone()
        for i in range(5):
            z = (self.freq_target_constant * z / z.mean()).clamp(0, 1)
        return z.cuda()

    @property
    def features(self):
        return dict(
            thresh_values=ft.FeaturesParam(
                self.thresh_values,
                feature_index=0,
                fptype="bias",
                reset_optim_on_resample=False,
            )
        )

    def register_freq_tracker(self, ft):
        self.freqs = ft
        return ft

    def diff(self, x, target) -> torch.Tensor:
        if self.cfg.log_diff:
            return (log(x + self.cfg.logeps) - log(target + self.cfg.logeps)) * target
        return x - target

    @torch.no_grad()
    def post_step_hook_with_cache(self, cache: cl.Cache):
        if self.freqs.freqs is None:
            return
        if cache._is_dead:
            return
        assert cache._ancestor.has.L0
        L0 = cache._ancestor.L0
        avg_freq = L0 / self.thresh_values.shape[0]
        diff = torch.zeros_like(self.freqs.freqs)
        zipf_diff = self.diff(self.freqs.freqs, self.freq_target)
        ### VVV zipf thing check
        uni_diff = self.diff(self.freqs.freqs, self.freq_target_constant)
        zdsign = zipf_diff.sign()
        diff = torch.where(
            zdsign == uni_diff.sign(),
            zdsign
            * torch.min(
                zipf_diff.abs(),
                uni_diff.abs(),
            ),
            0,
        )
        if self.cfg.stepclamp:
            diff.clamp_(-self.cfg.stepclamp, self.cfg.stepclamp)
        ### ^^^
        diff += self.diff(avg_freq, self.freq_target_constant) * self.cfg.l0_diff_mult
        self.thresh_value_momentum.lerp_(diff, 1 - self.cfg.momentum)
        step = self.thresh_value_momentum
        if self.cfg.stepclamp:
            step.clamp_(-self.cfg.stepclamp, self.cfg.stepclamp)
        if self.t < self.cfg.warmup_len:
            step *= self.t / self.cfg.warmup_len
        self.thresh_values += step * self.cfg.lr
        if self.cfg.decay_toward_mean:
            self.thresh_values.lerp_(
                self.thresh_values.mean(), self.cfg.decay_toward_mean * self.cfg.lr
            )
        self.thresh_values.relu_()
        self.t += 1
        ### zipf
        if self.t % 100 == 0 and self.t > 1000:
            r = min(1, (self.t) / 2_000) * self.cfg.zipf_r
            i = self.cfg.zipf_i
            self.freq_target = self.zipf(i=i, r=r)
            self.freq_target.clamp_(0, 1)


class DynamicThreshSAE(Architecture[DynamicZipfThreshConfig]):
    def setup(self):
        pass

    @model_prop
    def model(self):
        thrlu = Thresholder(self.init, self.cfg.thresh_cfg)
        return Seq(
            encoder=cl.ReuseForward(
                Seq(
                    **useif(self.cfg.pre_bias, pre_bias=self.init._decoder.sub_bias()),
                    lin=self.init.encoder,
                    pre_acts=PreActMetrics(),
                    nonlinearity=thrlu,
                    freqs=thrlu.register_freq_tracker(EMAFreqTracker()),
                )
            ),
            metrics=co.metrics.ActMetrics(),
            penalty=LinearDecayL1Penalty(
                begin=self.cfg.l1_decay_start,
                end=self.cfg.l1_decay_end,
                end_scale=self.cfg.l1_end_scale,
            ),
            decoder=ft.OrthogonalizeFeatureGrads(
                ft.NormFeatures(
                    self.init.decoder,
                ),
            ),
        )

    @loss_prop
    def L2_loss(self):
        return L2Loss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)
