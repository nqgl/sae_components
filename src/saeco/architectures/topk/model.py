import torch

import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl

from saeco.architectures.initialization.initializer import Initializer
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)

from saeco.components.hooks.clipgrad import ClipGrad
from saeco.core import Seq

from saeco.misc import useif
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.sweeps import SweepableConfig
from saeco.core.basic_ops import Sub

from saeco.components.ops.fnlambda import Lambda


class TopKConfig(SweepableConfig):
    pre_bias: bool = False


def TopK(k):
    k = int(k)

    def _topk(x):
        v, i = x.topk(k, dim=-1, sorted=False)
        return torch.zeros_like(x).scatter_(-1, i, v)

    return Lambda(_topk)


def topk_sae(
    init: Initializer,
    cfg: TopKConfig,
):
    model = Seq(
        encoder=Seq(
            **useif(cfg.pre_bias, pre_bias=Sub(init.decoder.bias)),
            lin=init.encoder,
            nonlinearity=nn.ReLU(),
            topk=TopK(init.l0_target),
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        null_penalty=co.LambdaPenalty(
            lambda x: torch.zeros(1)
        ),  # "no sparsity penalty"
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
    tr = TrainingRunner(cfg, model_fn=topk_sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    from .config import cfg, PROJECT
