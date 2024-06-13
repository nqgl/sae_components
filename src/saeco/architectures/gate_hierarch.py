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
    # init._encoder.bias = False
    init._encoder.add_wrapper(ReuseForward)
    # init._decoder.add_wrapper(
    # init._decoder.add_wrapper(
    # W_dec = ex
    # ft.NormFeatures

    #     ft.OrthogonalizeFeatureGrads
    BF = 4
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
    directions = ReuseForward(
        Seq(
            pre_bias=ReuseForward(Sub(init.b_dec)),
            enc=enc_mag,
            reshape=Lambda(
                lambda x: einops.rearrange(
                    x, "b (d1 d2) -> b d1 d2", d1=BF, d2=init.d_dict // BF
                )
            ),
            dec=init.decoder,
            join=Lambda(lambda x: x.sum(dim=1)),
        )
    )

    model_aux = Seq(
        Parallel(
            acts=Seq(
                encoder=enc_gate,
            ),
            directions=Seq(
                directions, Lambda(lambda x: x / x.norm(dim=-1, keepdim=True))
            ),
        )
    )

    # models
    gated_model = Seq(
        # pre_bias=ReuseForward(Sub(init.b_dec)),
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

    # losses
    gated_model = model_aux
    losses = dict(
        L2_loss=L2Loss(gated_model),
        L2_aux_loss=L2Loss(model_aux),
        sparsity_loss=SparsityPenaltyLoss(model_aux),
    )
    return [gated_model, model_aux], losses
