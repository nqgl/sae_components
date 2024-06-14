import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

from saeco.architectures.base import SAE
from saeco.architectures.initialization.initializer import Initializer
from saeco.components.ops.detach import Thresh
import saeco.core as cl
import saeco.core.module
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
from saeco.architectures.initialization.tools import (
    reused,
    weight,
    bias,
    mlp_layer,
    layer,
)

import saeco.components as co
from saeco.trainer.trainable import Trainable
import einops
from saeco.components.gated import HGated, Gated


def gate_two_weights(init: Initializer, detach=True, untied=True):
    # init._encoder.bias = False
    init._encoder.add_wrapper(ReuseForward)
    init._decoder.add_wrapper(ft.NormFeatures)
    init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)

    enc_mag = Seq(
        lin=init.encoder,
        nonlinearity=nn.ReLU(),
    )

    enc_gate = ReuseForward(
        Seq(
            **co.useif(
                detach,
                detach=Lambda(lambda x: x.detach()),
            ),
            lin=(init._encoder.make_new()),
            nonlinearity=nn.ReLU(),
        )
    )

    # models
    gated_model = Seq(
        pre_bias=ReuseForward(init._decoder.sub_bias()),
        encoder=cl.Parallel(
            magnitude=enc_mag,
            gate=Thresh(enc_gate),
        ).reduce(
            lambda x, y: x * y,
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        decoder=init.decoder,
    )

    model_aux = Seq(
        pre_bias=ReuseForward(init._decoder.sub_bias()),
        encoder=enc_gate,
        L1=L1Penalty(),
        freqs=EMAFreqTracker(),
        decoder=init._decoder.detached if detach else init.decoder,
    )

    # losses
    losses = dict(
        L2_loss=L2Loss(gated_model),
        L2_aux_loss=L2Loss(model_aux),
        sparsity_loss=SparsityPenaltyLoss(model_aux),
    )
    return [gated_model, model_aux], losses


class HierarchicalInitializer(Initializer):
    def __init__(self):
        super().__init__()
        self.branching_factor = 4


def hierarchical(init: Initializer, detach=True, untied=True):
    init._encoder.add_wrapper(ReuseForward)
    BF = 2**4
    tl_init = Initializer(init.d_data, init.d_dict // BF)
    init._decoder.bias = False

    enc_mag = Seq(
        lin=init.encoder,
        nonlinearity=nn.ReLU(),
    )

    enc_gate = ReuseForward(
        Seq(
            lin=(tl_init._encoder.make_new()),
            nonlinearity=nn.ReLU(),
        )
    )
    LL = BF
    HL = init.d_dict // BF

    def rearrange(x):
        er = einops.rearrange(
            x,
            "b (d1 d2) -> b d1 d2",
            d1=HL,
            d2=LL,
        )
        return er

    def mul(er, param):
        return einops.einsum(er, param, "b hl ll, hl ll d -> b hl d")

    sub_b_dec = ReuseForward(Sub(init.b_dec))
    add_b_dec = ReuseForward(Add(init.b_dec))
    directions = ReuseForward(  # There is no norming or orthogonalization here
        Seq(
            enc=enc_mag,
            metrics=co.metrics.Metrics(L0=co.metrics.L0(), L1=co.metrics.L1()),
            l1_shrink=Parallel(
                shrunk=Seq(
                    shrink=Lambda(lambda x: x / 10),
                    penalty=L1Penalty(),
                ),
                ret=cl.ops.Identity(),
            ).reduce(lambda x, y: y),
            mul=Parallel(
                rearranged=Lambda(
                    lambda x: einops.rearrange(
                        x,
                        "b (d1 d2) -> b d1 d2",
                        d1=HL,
                        d2=LL,
                    )
                ),
                param=nn.Parameter(torch.randn(HL, LL, init.d_data) * 0.001),
            ).reduce(
                lambda er, param: einops.einsum(er, param, "b hl ll, hl ll d -> b hl d")
            ),
        )
    )

    model_aux = Seq(
        pre_bias=ReuseForward(sub_b_dec),
        parallel=Parallel(
            enc_gate=Seq(
                gate=enc_gate,
                penalty=L1Penalty(),
            ),
            directions=Seq(
                directions, Lambda(lambda x: x / x.norm(dim=-1, keepdim=True))
            ),
        ).reduce(lambda x, y: einops.einsum(x, y, "b hl, b hl d -> b d")),
        post_bias=add_b_dec,
    )

    full_model = Seq(
        pre_bias=ReuseForward(sub_b_dec),
        parallel=Parallel(
            enc_gate=Seq(
                gate=Thresh(enc_gate),
                metrics=co.metrics.ActMetrics(),  # not quite right on the metrics but just want to get this going for now
            ),
            directions=directions,
        ).reduce(lambda x, y: einops.einsum(x, y, "b hl, b hl d -> b d")),
        post_bias=add_b_dec,
    )

    losses = dict(
        L2_loss=L2Loss(full_model),
        L2_aux_loss=L2Loss(model_aux),
        sparsity_loss=SparsityPenaltyLoss(model_aux, num_expeted=2),
    )

    return [full_model, model_aux], losses


def hierarchical_l1scale(init: Initializer, detach=True, untied=True):
    init._encoder.add_wrapper(ReuseForward)
    init._decoder.add_wrapper(ft.NormFeatures)
    init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
    BF = 2**4
    tl_init = Initializer(init.d_data, init.d_dict // BF)
    # init._decoder.bias = False

    enc_mag = Seq(
        lin=init.encoder,
        nonlinearity=nn.ReLU(),
    )

    enc_gate = Seq(
        lin=init._encoder.make_new(),
        nonlinearity=nn.ReLU(),
    )

    enc_hl = ReuseForward(
        Seq(
            lin=(tl_init.encoder),
            nonlinearity=nn.ReLU(),
        )
    )
    LL = BF
    HL = init.d_dict // BF

    def model(enc, penalties, metrics):
        return Seq(
            pre_bias=ReuseForward(init._decoder.sub_bias()),
            encoder=enc,
            freqs=EMAFreqTracker(),
            **penalties,
            metrics=metrics,
            decoder=init.decoder,
        )

    gated = Gated(gate=enc_gate, mag=enc_mag)
    hgated = HGated(hl=enc_hl, ll=gated, bf=BF)

    model_aux0 = model(
        gated.aux(),
        penalties=dict(l1=L1Penalty()),
        metrics=co.metrics.Metrics(L0=co.metrics.L0(), L1=co.metrics.L1()),
    )

    model_aux1 = model(
        hgated.aux(),
        penalties=dict(l1=L1Penalty()),
        metrics=co.metrics.Metrics(L0=co.metrics.L0(), L1=co.metrics.L1()),
    )

    model_full = model(
        hgated.full(),
        penalties={},
        metrics=co.metrics.ActMetrics(),
    )

    models = [model_full, model_aux0, model_aux1]

    losses = dict(
        L2_loss=L2Loss(model_full),
        L2_aux_loss0=L2Loss(model_aux0),
        L2_aux_loss1=L2Loss(model_aux1),
        sparsity_loss=SparsityPenaltyLoss(model_aux0, num_expeted=2),
    )
    return models, losses


# if __name__ == "__main__":
#     from saeco.trainer.runner import TrainingRunner
