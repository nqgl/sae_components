import torch

import torch.nn as nn
import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.architectures.initialization.initializer import Initializer
from saeco.components import (
    L1Penalty,
    LinearDecayL1Penalty,
    EMAFreqTracker,
    L2Loss,
    FreqTracker,
    SparsityPenaltyLoss,
)

from saeco.components.hooks.clipgrad import ClipGrad
from saeco.core import Seq

from saeco.misc import useif
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.sweeps import SweepableConfig, Swept
from pydantic import Field

# torch.backends.cudnn.benchmark = True


class ThreshConfig(SweepableConfig):
    # eps: float = Swept(1e1)
    logeps: float = 3e-7
    lr: float = 0.03  # Swept(0.1, 0.01)
    signstep: float = 0
    warmup_len: float = 5_000
    momentum: float = 0.9
    stepclamp: None | float = 1e-1
    sign_switch_decay_mul: float = 1.0
    end_scale: float = Swept(0, 0.01)
    # freq_ratios: None | float = 1
    decay_toward_mean: float = Swept(
        1.5, 2.0, 1, 0.7, 3
    )  ###Swept(1, 6e-1, 3e-1, 1e-1, 3e-2, 1e-3)  # Swept(
    # 1e-2, 1e-3, 1e-4
    # )  # Swept(0.003, 0.001, 0.0003, 0.0001)
    log_diff: bool = True
    l0_diff_mult: float = 30
    initial_value: float = 0
    # l0_targ_lin: bool = Swept(True, False)

    # min_freq_ratio: float | None = Swept(3, 10)
    # max_freq_ratio: float | None = Swept(3, 10)
    # @property
    # def min_freq_ratio(self):
    #     return self.freq_ratios

    # @property
    # def max_freq_ratio(self):
    #     return self.freq_ratios


class Config(SweepableConfig):
    thresh_cfg: ThreshConfig = Field(default_factory=ThreshConfig)
    pre_bias: bool = False


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
        # self.thresh_values = torch.zeros(init.d_dict)
        self.nonlinearity = nn.ReLU()
        self.target = init.l0_target
        self.freqs: co.FreqTracker = None
        self.freq_target = init.l0_target / init.d_dict
        self.t = 0
        self.prev_cache = None

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        x = cache(self).nonlinearity(x)
        self.prev_cache = cache
        return torch.where(
            x > self.thresh_values.unsqueeze(0),
            x,
            0,
        )

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
    def post_step_hook(self):
        if self.freqs.freqs is None:
            return
        if self.prev_cache._is_dead:
            return
        assert self.prev_cache._ancestor.has.L0
        L0 = self.prev_cache._ancestor.L0
        avg_freq = L0 / self.thresh_values.shape[0]
        diff = torch.zeros_like(self.freqs.freqs)
        diff += self.diff(self.freqs.freqs, self.freq_target)
        diff += self.diff(avg_freq, self.freq_target) * self.cfg.l0_diff_mult
        self.thresh_value_momentum.lerp_(diff, 1 - self.cfg.momentum)
        step = self.thresh_value_momentum
        if self.cfg.stepclamp:
            step.clamp_(-self.cfg.stepclamp, self.cfg.stepclamp)
        if self.t < self.cfg.warmup_len:
            step *= self.t / self.cfg.warmup_len
        self.thresh_values += step * self.cfg.lr
        self.thresh_values.relu_()
        if self.cfg.decay_toward_mean:
            self.thresh_values.lerp_(
                self.thresh_values.mean(), self.cfg.decay_toward_mean * self.cfg.lr
            )
        self.t += 1


def sae(
    init: Initializer,
    cfg: Config,
):
    thrlu = Thresholder(init, cfg.thresh_cfg)
    model = Seq(
        encoder=Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            lin=init.encoder,
            nonlinearity=thrlu,
        ),
        freqs=thrlu.register_freq_tracker(EMAFreqTracker()),
        metrics=co.metrics.ActMetrics(),
        penalty=LinearDecayL1Penalty(
            begin=5_000, end=25_000, end_scale=cfg.thresh_cfg.end_scale
        ),
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


from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


def run(cfg):
    if (
        cfg.train_cfg.coeffs["sparsity_loss"] == 0
        and cfg.arch_cfg.thresh_cfg.end_scale != 0
    ):
        raise ValueError("skipping redundant sweep")
    # elif (
    #     cfg.arch_cfg.thresh_cfg.freq_ratios is None
    #     and cfg.arch_cfg.thresh_cfg.decay_toward_mean != 1e-2
    # ):
    #     raise ValueError("skipping redundant sweep")
    tr = TrainingRunner(cfg, model_fn=sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    from .config import cfg, PROJECT
