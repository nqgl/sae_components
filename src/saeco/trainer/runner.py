from saeco.architectures.gate_two_weights import gate_two_weights
from saeco.data.sc.dataset import DataConfig, SplitConfig
from transformer_lens import HookedTransformer

from saeco.trainer.trainable import Trainable

from saeco.trainer.trainer import Trainer, TrainConfig
from typing import Generic, TypeVar
from saeco.architectures.initialization.geo_med import getmed, getmean
from saeco.architectures.gate_hierarch import (
    hierarchical_l1scale,
    hierarchical_softaux,
    HierarchicalSoftAuxConfig,
    HGatesConfig,
)
from saeco.architectures.vanilla_tests import (
    basic_vanilla_sae_lin,
)
from saeco.architectures.deep.deep import deep_sae, resid_deep_sae
from saeco.architectures.deep.deep_resid_gated import (
    deep_resid_gated,
    deep_resid_gated2,
    deep_resid_gated2_wider,
    deep_resid_gated2_deeper,
    deep_resid_gated2_deeper_still,
    deep_resid_gated2_wider,
    deep_resid_gated2_wider2,
)
from saeco.architectures.deep.catseq import deep_catseq, deep_catseq_resid
import wandb
from saeco.architectures.remax import (
    remax_sae,
    remax1_sae,
    remaxk_sae,
    remaxkv_sae,
    remaxkB_sae,
    remaxkvB_sae,
)
from saeco.architectures.topk import topk_sae
import torch
import sys
from saeco.architectures.initialization.initializer import Initializer
from saeco.trainer.normalizers import (
    ConstL2Normalizer,
    Normalized,
    Normalizer,
    L2Normalizer,
    NORMALIZERS,
)
from saeco.misc.lazy import lazyprop, defer_to_and_set
from saeco.sweeps import SweepableConfig
from pydantic import BaseModel, Field


class SAEConfig(SweepableConfig):
    d_data: int = 768
    dict_mult: int = 8
    normalizer: str = "ConstL2Normalizer"

    @lazyprop
    def d_dict(self):
        return self.d_data * self.dict_mult

    @d_dict.setter
    def d_dict(self, value):
        assert self.dict_mult is None
        setattr(self, "_d_dict", value)


T = TypeVar("T", bound=SweepableConfig)


class RunConfig(SweepableConfig, Generic[T]):
    train_cfg: TrainConfig
    arch_cfg: T
    sae_cfg: SAEConfig = Field(default_factory=SAEConfig)


class TrainingRunner:
    def __init__(self, cfg: RunConfig, model_fn):
        self.cfg = cfg
        self.model_fn = model_fn
        self._models = None
        self._losses = None

    @lazyprop
    def model_name(self):
        return self.model_fn.__name__

    @lazyprop
    def name(self):
        return f"{self.model_name}{self.cfg.train_cfg.lr}"

    @lazyprop
    def buf(self) -> iter:
        return iter(self.cfg.train_cfg.data_cfg.get_databuffer())

    @lazyprop
    def initializer(self) -> Initializer:
        return Initializer(
            self.cfg.sae_cfg.d_data,
            dict_mult=self.cfg.sae_cfg.dict_mult,
            l0_target=self.cfg.train_cfg.l0_target,
            median=getmean(buf=self.buf, normalizer=self.normalizer),
            # median=getmed(buf=self.buf, normalizer=self.normalizer),
            # weight_scale=2,
        )

    @defer_to_and_set("_model_fn_output")
    def get_model_fn_output(self):
        assert self._models is self._losses is None
        if self.cfg.arch_cfg is None:
            return self.model_fn(self.initializer)
        return self.model_fn(self.initializer, self.cfg.arch_cfg)

    @lazyprop
    def models(self):
        models, losses = self.get_model_fn_output()
        return models

    @lazyprop
    def losses(self):
        models, losses = self.get_model_fn_output()
        return losses

    @lazyprop
    def trainable(self):
        return Trainable(self.models, self.losses, normalizer=self.normalizer).cuda()

    @lazyprop
    def normalizer(self):
        normalizer = NORMALIZERS[self.cfg.sae_cfg.normalizer]()
        normalizer.prime_normalizer(self.buf)
        return normalizer

    @normalizer.setter
    def normalizer(self, value):
        self._normalizer = value
        self._normalizer.prime_normalizer(self.buf)

    @lazyprop
    def trainer(self):
        trainer = Trainer(self.cfg.train_cfg, self.trainable, namestuff=self.name)
        trainer.post_step()
        return trainer


def main():
    l0_target = 45
    PROJECT = "nn.Linear Check"
    tcfg = TrainConfig(
        l0_target=l0_target,
        coeffs={
            "sparsity_loss": 3e-4,
            "L2_loss": 1,
        },
        lr=1e-3,
        use_autocast=True,
        wandb_cfg=dict(project=PROJECT),
        l0_target_adjustment_size=0.001,
        batch_size=2048,
        use_lars=True,
        betas=(0.9, 0.99),
    )
    cfg = RunConfig(
        train_cfg=tcfg,
        arch_cfg=HierarchicalSoftAuxConfig(
            hgates=HGatesConfig(
                l1_scale_base=1,
                num_levels=2,
                BF=2**4,
                untied=False,
                classic=True,
                penalize_inside_gate=False,
                target_hierarchical_l0_ratio=0.5,
                relu_gate_encoders=False,
            )
        ),
        sae_cfg=SAEConfig(normalizer="L2Normalizer"),
    )

    tr = TrainingRunner(cfg, hierarchical_softaux)

    tr.normalizer = ConstL2Normalizer()
    tr.trainer
    tr.trainer.train()


if __name__ == "__main__":
    main()


def norm_consts(mean, std, geo_med, std_from_med, elementwise_std=False):
    return mean, std, geo_med, std_from_med


"""
if we have a non-elementwise normalizer NE and an elementwise normalizer E
is E(x) = NE(E(x))?

LinLearnedLN(x) = (x - mean) / estd


also is LLLN(x) = T^-1(LLLN(T(x))) for linear transformations T?

"""
