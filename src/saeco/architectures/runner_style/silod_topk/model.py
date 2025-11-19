import torch
import torch.nn as nn
from saeco.architectures.topk.TopK import NormalizedResidL2Loss, TopK, TopKDead

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.components import (
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)
from saeco.components.ops.fnlambda import Lambda
from saeco.core import Seq
from saeco.core.basic_ops import Sub
from saeco.initializer import Initializer
from saeco.misc import useif
from saeco.sweeps import SweepableConfig


class SilodTopKConfig(SweepableConfig):
    pre_bias: bool = False
    num_silos: int = 4
    aux_top_k: int = 512
    dead_threshold: float = 1e-5
    skew: float = 1


def silod_topk_sae(
    init: Initializer,
    cfg: SilodTopKConfig,
):
    silo_targets = [init.l0_target // cfg.num_silos] * (cfg.num_silos)
    for i in range(init.l0_target % cfg.num_silos):
        silo_targets[i] += 1

    silo_sizes = torch.linspace(1, cfg.skew, cfg.num_silos)
    silo_sizes = (silo_sizes * init.d_dict / silo_sizes.sum()).round().int()
    diff = init.d_dict - silo_sizes.sum()
    silo_sizes += diff // cfg.num_silos
    silo_sizes[: diff % cfg.num_silos] += 1
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
            **useif(
                cfg.num_silos != 1,
                silotopk=Seq(
                    split=Lambda(lambda x: torch.split(x, silo_sizes.tolist(), dim=-1)),
                    router=cl.Router(*[TopK(k) for k in silo_targets]).reduce(
                        lambda *x: torch.cat(x, dim=-1)
                    ),
                ),
            ),
            **useif(
                cfg.num_silos == 1,
                topk=TopK(init.l0_target),
            ),
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
    tr = TrainingRunner(cfg, model_fn=silod_topk_sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    pass
