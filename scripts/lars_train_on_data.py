# %%


from saeco.data.sc.dataset import DataConfig, SplitConfig
from transformer_lens import HookedTransformer

from saeco.trainer.trainable import Trainable

gpt2 = HookedTransformer.from_pretrained("gpt2")
BATCH_SIZE = 4096


from saeco.architectures.gated import gated_sae, gated_sae_no_detach
from saeco.architectures.vanilla_tests import (
    vanilla_sae,
    basic_vanilla_sae_lin,
    basic_vanilla_sae_lin_no_orth,
)
from saeco.architectures.vanilla.unshrink import rescaling_vanilla
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
from saeco.architectures.topk import topk_sae
from saeco.architectures.remax import remax_sae, remax1_sae
from saeco.architectures.deep.catseq import deep_catseq, deep_catseq_resid
import wandb
import torch
from saeco.architectures.initialization.initializer import Initializer

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

from saeco.architectures.initialization.tools import bias
from saeco.architectures.initialization.tools import weight
from saeco.architectures.initialization.tools import mlp_layer
from saeco.components.ops.detach import Thresh
import saeco.core as cl
from saeco.core.collections.parallel import Parallel
from saeco.components import (
    Penalty,
    L1Penalty,
    FreqTracked,
    EMAFreqTracker,
    FreqTracker,
    ResampledLayer,
    Loss,
    L2Loss,
    SparsityPenaltyLoss,
    SAECache,
)

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Add, MatMul, Sub, Mul
from typing import Optional
from saeco.components.ops.fnlambda import Lambda
from saeco.core.reused_forward import ReuseForward, ReuseCache
from saeco.core import Seq
import saeco.components.features.features as ft
import saeco.components as co


from saeco.core.collections.seq import ResidualSeq
from saeco.trainer.trainable import Trainable

from saeco.trainer.normalizers import (
    ConstL2Normalizer,
    Normalized,
    Normalizer,
    L2Normalizer,
)

import saeco.architectures.vanilla_tests as vanilla

from torchlars import LARS


# %%
PROJECT = "nn.Linear Check"


def train_lars(model_fn, l0_target=45, lr=3e-4):
    from saeco.trainer.trainer import Trainer, TrainConfig

    name = "(lars)" + model_fn.__name__
    models, losses = model_fn(Initializer(768, dict_mult=8, l0_target=l0_target))

    cfg = TrainConfig(
        l0_target=l0_target,
        coeffs={
            "sparsity_loss": 2e-3 if l0_target is None else 14e-4,
        },
        lr=lr,
        use_autocast=True,
        wandb_cfg=dict(project=PROJECT),
        l0_target_adjustment_size=0.001,
    )
    trainable = Trainable(models, losses, normalizer=L2Normalizer()).cuda()
    trainer = Trainer(cfg, trainable, namestuff=name + f"_{lr:.0e}")
    buf = iter(trainer.get_databuffer())
    trainable.normalizer.prime_normalizer(buf)
    trainer.post_step()
    optim = torch.optim.RAdam(trainable.parameters(), lr=lr, betas=(0.9, 0.99))
    trainer.optim = LARS(optim)
    trainer.train(buf)


# train_lars(basic_vanilla_sae_lin_no_orth, l0_target=45, lr=1e-3)
# train_lars(basic_vanilla_sae_lin, l0_target=45, lr=1e-3)
train_lars(vanilla.classed_basic_vanilla_sae_lin, l0_target=45, lr=3e-4)
train_lars(remax1_sae, l0_target=45, lr=1e-3)

train_lars(rescaling_vanilla, l0_target=20, lr=1e-3)
train_lars(gated_sae, l0_target=20, lr=1e-3)
train_lars(topk_sae, l0_target=20, lr=1e-3)
train_lars(remax1_sae, l0_target=45, lr=1e-3)


train_lars(gated_sae_no_detach, l0_target=45, lr=1e-3)


train_lars(deep_resid_gated2, l0_target=45, lr=1e-3)
