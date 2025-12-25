import torch
import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.components import (
    EMAFreqTracker,
    L2Loss,
    LambdaPenalty,
    Loss,
    SAECache,
    SparsityPenaltyLoss,
)
from saeco.components.ops.fnlambda import Lambda
from saeco.components.ops.nonlinearities.softlu_remax import (
    ReMax,
    ReMaxK,
    ReMaxKv,
)
from saeco.core import Seq

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Add, Sub
from saeco.initializer import Initializer


class L2RescaledLoss(Loss):
    def loss(self, x, y, y_pred, cache: SAECache):
        with torch.no_grad():
            yn = y.norm(dim=1, keepdim=True)
            y_predn = y_pred.norm(dim=1, keepdim=True)
            scaling = 0.99 * yn / y_predn
        return ((y - y_pred * scaling) ** 2).mean()


class Rescale(cl.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        y_pred = self.module(x, cache=cache, **kwargs)
        if cache._ancestor.has.scale and cache._ancestor.scale:
            with torch.no_grad():
                scale = x.norm(dim=1, keepdim=True) / y_pred.norm(dim=1, keepdim=True)
        else:
            return y_pred
        return y_pred * scale


class RangeLimited(cl.Module):
    def __init__(self, param, min, max):
        super().__init__()
        self.param = param
        self.min = min
        self.max = max

    @torch.no_grad()
    def post_step_hook(self):
        self.param.data.clamp_(self.min, self.max)

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return self.param


def remax_sae(init: Initializer, remax_fn=None, basescale=50, **remax_kwargs):
    scalenet = Seq(
        relu=nn.ReLU(),
        scale=nn.Linear(init.d_dict, 1),
        nonlinearity=Lambda(lambda x: torch.relu(x) + init.d_data**0.5),
    )

    init._decoder.bias = False
    # b_dec = init.data_bias()
    scaleparam = nn.Parameter(torch.ones(1))
    model = Seq(
        pre_bias=Sub(init.b_dec),
        resc=Rescale(
            Seq(
                encoder=Seq(
                    lin=init.encoder,
                    nonlinearity=(remax_fn or ReMax)(scale=basescale, **remax_kwargs),
                    scale=cl.ops.Mul(
                        RangeLimited(scaleparam, 0.9, 1.05),
                    ),
                    # scale=cl.ops.MulParallel(
                    #     identity=cl.ops.Identity(),
                    #     scale_exp=Seq(
                    #         _support_parameters=True,
                    #         scaleparam=scaleparam,
                    #         exp=Lambda(torch.exp),
                    #     ),
                    # ),
                    # scale=cl.ops.MulParallel(
                    #     mag=scalenet,
                    #     input=cl.ops.Identity(),
                    # ),
                ),
                freqs=EMAFreqTracker(),
                metrics=co.metrics.ActMetrics(),
                # null_penalty=LambdaPenalty(lambda x: 0),  # "no sparsity penalty"
                sparsity_scale_penalty=LambdaPenalty(lambda x: torch.zeros(1)),
                decoder=ft.OrthogonalizeFeatureGrads(
                    ft.NormFeatures(
                        init.decoder,
                    ),
                ),
            )
        ),
        post_bias=Add(init.b_dec),
    )

    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    # losses = dict(
    #     L2_rescaled_loss=L2RescaledLoss(model),
    #     sparsity_loss=SparsityPenaltyLoss(model),
    # )
    return models, losses


def remax1_sae(init: Initializer, **kwargs):
    return remax_sae(init, remax1=True, **kwargs)


def remaxk_sae(init: Initializer, **kwargs):
    return remax_sae(init, remax_fn=ReMaxK, norm=1, k=init.l0_target, **kwargs)


def remaxkv_sae(init: Initializer, **kwargs):
    return remax_sae(init, remax_fn=ReMaxKv, k=init.l0_target, **kwargs)


def remaxkB_sae(init: Initializer, **kwargs):
    return remax_sae(init, remax_fn=ReMaxK, norm=1, k=init.l0_target, b=True, **kwargs)


def remaxkvB_sae(init: Initializer, **kwargs):
    return remax_sae(init, remax_fn=ReMaxKv, k=init.l0_target, b=True, **kwargs)
