import torch

import torch.nn as nn

from saeco.architectures.topk.TopK import NormalizedResidL2Loss, TopK, TopKDead
import saeco.components as co
import saeco.components.features.features as ft
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

from saeco.misc import useif
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.sweeps import SweepableConfig
from saeco.core.basic_ops import Sub


class TopKConfig(SweepableConfig):
    pre_bias: bool = False
    aux_top_k: int = 512
    dead_threshold: float = 1e-5


def topk_sae(
    init: Initializer,
    cfg: TopKConfig,
):
    encoder = cl.ReuseForward(
        Seq(
            **useif(cfg.pre_bias, pre_bias=Sub(init.decoder.bias)),
            lin=init.encoder,
            nonlinearity=nn.ReLU(),
        )
    )
    freq_tracker = EMAFreqTracker(0.999)
    decoder = ft.OrthogonalizeFeatureGrads(
        ft.NormFeatures(
            init.decoder,
        ),
    )
    model = cl.ReuseForward(
        Seq(
            encoder=encoder,
            topk=TopK(init.l0_target),
            # freqs=freq_tracker,
            freqs=freq_tracker,
            metrics=co.metrics.ActMetrics(),
            null_penalty=co.LambdaPenalty(
                lambda x: torch.zeros(1)
            ),  # "no sparsity penalty"
            decoder=decoder,
        )
    )

    aux_model = cl.Parallel(
        main_model=model,
        aux_model=Seq(
            encoder=encoder,
            topk=TopKDead(cfg.aux_top_k, freq_tracker, cfg.dead_threshold),
            decoder=init._decoder.biasdetached,
        ),
    ).reduce(lambda main, aux: main.detach() + aux)

    models = [model, aux_model]
    losses = dict(
        L2_loss=L2Loss(model),
        L2_aux_loss=NormalizedResidL2Loss(aux_model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return models, losses


from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


def run(cfg):
    if cfg.arch_cfg.aux_top_k == 0:
        assert (
            cfg.train_cfg.coeffs["L2_aux_loss"] == 1
            and cfg.arch_cfg.dead_threshold == 1e-5
        ), "skipping redundant sweep"
    if cfg.arch_cfg.dead_threshold == 3e-5:
        assert (
            cfg.train_cfg.raw_schedule_cfg.resample_period == 100_000
        ), "skipping redundant sweep"
    tr = TrainingRunner(cfg, model_fn=topk_sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    from .config import cfg, PROJECT
