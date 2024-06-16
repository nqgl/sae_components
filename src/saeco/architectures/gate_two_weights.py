import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.architectures.initialization.initializer import Initializer
from saeco.components import EMAFreqTracker, L1Penalty, L2Loss, SparsityPenaltyLoss
from saeco.components.ops.detach import Thresh
from saeco.components.ops.fnlambda import Lambda
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward


import torch.nn as nn

import saeco.misc.utils


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
            **saeco.misc.utils.useif(
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
