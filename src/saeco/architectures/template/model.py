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


class Config(SweepableConfig):
    pre_bias: bool = False
    clip_grad: float | None = 1


def sae(
    init: Initializer,
    cfg: Config,
):
    model = ...
    models = [model, ...]
    losses = {...}
    return models, losses
