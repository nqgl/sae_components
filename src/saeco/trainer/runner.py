# %%

from saeco.architectures.gate_two_weights import gate_two_weights
from saeco.data.sc.dataset import DataConfig, SplitConfig
from transformer_lens import HookedTransformer

from saeco.trainer.trainable import Trainable

gpt2 = HookedTransformer.from_pretrained("gpt2")
BATCH_SIZE = 4096
from saeco.trainer.trainer import Trainer, TrainConfig

from saeco.architectures.initialization.geo_med import getmed, getmean
from saeco.architectures.gated import gated_sae, gated_sae_no_detach
from saeco.architectures.gate_hierarch import (
    hierarchical_l1scale,
)
from saeco.architectures.vanilla_tests import (
    vanilla_sae,
    basic_vanilla_sae,
    basic_vanilla_sae_lin,
    basic_vanilla_sae_lin_no_orth,
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
)
from saeco.misc.lazy import lazyprop, defer_to_and_set


class TrainingRunner:
    def __init__(self, cfg, model_fn):
        self.cfg = cfg
        self.model_fn = model_fn
        self._models = None
        self._losses = None

    @lazyprop
    def model_name(self):
        return self.model_fn.__name__

    @lazyprop
    def name(self):
        return f"{self.model_name}{self.cfg.lr}"

    @lazyprop
    def buf(self) -> iter:
        return iter(self.cfg.data_cfg.get_databuffer())

    @lazyprop
    def initializer(self) -> Initializer:
        return Initializer(
            768,
            dict_mult=8,
            l0_target=self.cfg.l0_target,
            median=getmed(buf=self.buf, normalizer=self.normalizer),
            # weight_scale=2,
        )

    @defer_to_and_set("_model_fn_output")
    def get_model_fn_output(self):
        assert self._models is self._losses is None
        return self.model_fn(self.initializer)

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
        normalizer = ConstL2Normalizer()
        normalizer.prime_normalizer(self.buf)
        return normalizer

    @normalizer.setter
    def normalizer(self, value):
        self._normalizer = value
        self._normalizer.prime_normalizer(self.buf)

    @lazyprop
    def trainer(self):
        trainer = Trainer(self.cfg, self.trainable, namestuff=self.name)
        trainer.post_step()
        return trainer


l0_target = 45
PROJECT = "nn.Linear Check"
cfg = TrainConfig(
    l0_target=l0_target,
    coeffs={
        "sparsity_loss": 2e-3 if l0_target is None else 8e-4,
    },
    lr=3e-4,
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0001,
    batch_size=4096,
    use_lars=True,
    betas=(0.9, 0.99),
)
tr = TrainingRunner(cfg, hierarchical_l1scale)

tr.normalizer = ConstL2Normalizer()
tr.trainer
# %%
tr.trainer.train()
# %%


def norm_consts(mean, std, geo_med, std_from_med, elementwise_std=False):
    return mean, std, geo_med, std_from_med


"""
if we have a non-elementwise normalizer NE and an elementwise normalizer E
is E(x) = NE(E(x))?

LinLearnedLN(x) = (x - mean) / estd


also is LLLN(x) = T^-1(LLLN(T(x))) for linear transformations T?

"""
