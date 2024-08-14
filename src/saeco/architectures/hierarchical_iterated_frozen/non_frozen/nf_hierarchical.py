from typing import Iterator
import torch

import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
from saeco.components.hierarchical import get_acts_only
from saeco.components.hierarchical import ActsGateSubstitutor
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
from saeco.components.resampling.freq_tracker.freq_tracker import get_freq_trackers
from saeco.misc import useif
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.sweeps import SweepableConfig
from saeco.components.threshgate import (
    ThreshGate,
    SoftenedThreshGate,
)

# TODO
# add feature to specify dominant encoder
# oh, we've actually got to just fully disable resampling on the earlier layers

# TODO
# do an "update optim" step when incrementing the level

from saeco.sweeps import Swept
from saeco.architectures.anth_update.model import anth_update_model, AnthUpdateConfig

from saeco.architectures.topk.model import topk_sae, TopKConfig
from pydantic import BaseModel
from typing import Callable


class LLModelSpec(BaseModel):
    # name: str
    config: type
    model_fn: Callable
    # site_name: str
    write_site: str
    read_site: str
    pre_site: str = None


specs = dict(
    topk=LLModelSpec(
        # name="topk",
        config=TopKConfig,
        model_fn=topk_sae,
        write_site="nonlinearity",
        read_site="encoder",
    ),
    anth_update=LLModelSpec(
        # name="anth_update",
        config=AnthUpdateConfig,
        model_fn=anth_update_model,
        write_site="encoder",
        read_site="encoder",
        pre_site="linear",
    ),
)


class HSAEConfig(SweepableConfig):
    pre_bias: bool = False
    branching_factor: int = 32
    num_levels: int = 2
    # acts_name: str = "encoder"
    ll_model: str = "anth_update"
    l0_target_ratio: float = 1
    residual: bool = True

    @property
    def ll_model_fn(self):
        return specs[self.ll_model].model_fn

    @property
    def ll_acts_read_site(self):
        return specs[self.ll_model].read_site

    @property
    def ll_acts_write_site(self):
        return specs[self.ll_model].write_site

    @property
    def ll_cfg(self):
        return specs[self.ll_model].config


class HSAELayerConfig(SweepableConfig): ...


def indexmixed(x, il, bf):
    # note to self for future:
    # hl2ll([1,2,3], 2) = [1, 1, 2, 2, 3, 3]
    bfs = [bf**e for e in range(len(il))]


class HSAELayer(cl.Module):
    def __init__(self, cfg: HSAEConfig, models, losses):
        super().__init__()
        self.cfg = cfg
        # self.layer_cfg = layer_cfg
        # self.encode = Seq(
        #     encoder=init.encoder,
        #     relu=nn.ReLU(),
        # )
        assert len(models) == 1
        self.model = models[0]
        self.models = models
        self.losses = losses
        # # self.l1 = L1PenaltyScaledByDecoderNorm()
        # self.decode = init.decoder
        self.substitutor = ActsGateSubstitutor(
            self.model, ll_acts_key=cfg.ll_acts_write_site, bf=cfg.branching_factor
        )

    def forward(self, x, *, cache: cl.Cache, prev_acts: torch.Tensor):
        # basic_acts = cache(self).encode(x)
        # cache.acts = (
        #     acts := gate_ll_acts(basic_acts, prev_acts, self.cfg.branching_factor)
        # )
        return cache(self).substitutor(x, prev_acts=prev_acts)

    def get_acts(self, x, *, cache: cl.Cache, prev_acts: torch.Tensor):
        # return cache(self).encode(x)
        acts = get_acts_only(self.substitutor, x, cache=cache, prev_acts=prev_acts)
        return acts


from saeco.components.features.features_param import get_featuresparams


class HSAE(cl.Module):
    def __init__(self, init: Initializer, cfg: HSAEConfig):
        super().__init__()
        self.cfg = cfg
        self.l0s = [
            init.l0_target * cfg.l0_target_ratio**i for i in range(cfg.num_levels)
        ]
        if self.cfg.residual:
            self.l0s = [init.l0_target * l0 / sum(self.l0s) for l0 in self.l0s]
        assert not any([l0 < 1 for l0 in self.l0s])
        inits = [
            init.get_hl_initializer(
                cfg.branching_factor**i,
                l0_target=self.l0s[i],
            )
            for i in reversed(range(cfg.num_levels))
        ]
        # self.all_models = []
        # self.all_losses = []
        # for i in inits:
        #     m, l =
        #     self.all_models.append(m)
        #     self.all_losses.append(l)

        self.layers = nn.ModuleList(
            [
                HSAELayer(cfg=cfg, models=models, losses=losses)
                for models, losses in [cfg.ll_model_fn(i, cfg.ll_cfg()) for i in inits]
            ]
        )
        self.current_level = 0
        self.increment_level(0)

    def forward(self, x, *, cache: cl.Cache):
        if not self.cfg.residual:
            acts = None
            for i in range(self.current_level):
                acts = cache(self).layers[i].get_acts(x, prev_acts=acts)
            return cache(self).layers[self.current_level](x, prev_acts=acts)
        else:
            # cache = cache.clone()
            cache.gated_acts = ...
            acts = None
            x_pred = 0
            for i in range(self.current_level + 1):
                cc = cache.clone() if i < self.current_level else cache

                x_pred += cc(self).layers[i](x - x_pred, prev_acts=acts)
                acts = cc(self).layers[i]._cache["substitutor"].gated_acts
            return x_pred

    @property
    def losses(self):
        return dict(
            L2_loss=L2Loss(self),
            sparsity_loss=SparsityPenaltyLoss(self),
        )

    @property
    def models(self):
        return [self]

    def increment_level(self, n=None):
        n = n if n is not None else self.current_level + 1
        if n >= len(self.layers):
            raise ValueError("Cannot increment level beyond maximum")
        self.current_level = n

        # for layer in self.layers:

        #     for name, group in layer.named_parameters():
        #         group.requires_grad = False
        #     for param in get_featuresparams(layer):
        #         param.resampled = False  # TODO fix this behavior, this does NOT work how you would want it to
        #     for ft in get_freq_trackers(layer):
        #         ft.is_active = False
        # for name, group in self.layers[n].named_parameters():
        #     group.requires_grad = True
        # for param in get_featuresparams(self.layers[n]):
        #     param.resampled = True
        # for ft in get_freq_trackers(self.layers[n]):
        #     ft.is_active = True

    @property
    def current_l0_target(self):
        return self.l0s[self.current_level]

    # def train(self, mode: bool = True):
    #     super().train(mode)


# def model_fn(
#     init: Initializer,
#     cfg: HSAEConfig,
# ):
#     model = HSAE(init, cfg)

#     return models, losses


def run(cfg):
    from saeco.trainer.runner import TrainingRunner

    def hsaef():
        pass

    tr = TrainingRunner(cfg, model_fn=hsaef)
    init = tr.initializer
    hsae = HSAE(init, cfg.arch_cfg)
    prev_t = 0
    initial_sparsity = cfg.train_cfg.coeffs["sparsity_loss"]
    m = 27 / 0.9
    while True:
        cfg.train_cfg.l0_target = hsae.current_l0_target
        cfg.train_cfg.coeffs["sparsity_loss"] = initial_sparsity

        tr = TrainingRunner(cfg, model_fn=hsaef)
        init = tr.initializer

        tr._models = hsae.models
        tr._losses = hsae.losses

        trainer = tr.trainer
        trainer.log_t_offset = prev_t
        print("start next")
        trainer.train()
        print("increment level")
        hsae.increment_level()
        print("incremented level")
        prev_t = trainer.t + trainer.log_t_offset


from saeco.sweeps import do_sweep

if __name__ == "__main__":
    do_sweep(True)
else:
    from .config import cfg, PROJECT
