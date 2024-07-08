import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float
import saeco.core as cl
import saeco.components as cc

# from saeco.components import (
#     Penalty,
#     L1Penalty,
#     FreqTracked,
#     EMAFreqTracker,
#     FreqTracker,
#     ResampledLayer,
#     Loss,
#     L2Loss,
#     SparsityPenalty,
#     SAECache,
#     NegBias,
# )


class Resampled: ...
