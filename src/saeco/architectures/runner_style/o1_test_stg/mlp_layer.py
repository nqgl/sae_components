import torch.nn as nn
from saeco.architectures.sweep_tg.other_lin import OtherLinear

import saeco.components.features.features as ft
from saeco.core import Seq


def mlp_layer(
    d_in,
    d_hidden,
    d_out=None,
    nonlinearity=nn.LeakyReLU,
    normalize=False,
    no_resample=False,
):
    d_out = d_out or d_in
    if nonlinearity is nn.PReLU:
        nonlinearity = nn.PReLU(d_hidden).cuda()
    if isinstance(nonlinearity, type):
        nonlinearity = nonlinearity()
    proj_in = OtherLinear(nn.Linear(d_in, d_hidden), weight_param_index=1)
    proj_in.features["bias"].resampled = False
    proj_out = OtherLinear(nn.Linear(d_hidden, d_out), weight_param_index=1)
    proj_out.features["weight"].resampled = False
    if no_resample:
        proj_out.features["bias"].resampled = False
        proj_in.features["weight"].resampled = False

    if normalize:
        proj_in = ft.NormFeatures(proj_in, index="weight", ord=2, max_only=True)

        proj_out = ft.NormFeatures(proj_out, index="weight", ord=2, max_only=True)

    return Seq(proj_in=proj_in, nonlinearity=nonlinearity, proj_out=proj_out)
