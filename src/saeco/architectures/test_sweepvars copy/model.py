import torch

import torch.nn as nn

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


class Config(SweepableConfig):
    a: int
    b: int
    a_b: int


from saeco.sweeps import do_sweep


def run(cfg):
    print(cfg)


if __name__ == "__main__":
    do_sweep(True, "new run")
else:
    from .config import cfg, PROJECT
