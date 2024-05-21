import torch
import torch.nn as nn
import sae_components.core as cl
from sae_components.components.sae_cache import SAECache
from torch import Tensor
from jaxtyping import Float

from typing import Tuple, Any, Union, Optional, List
from sae_components.components.resampling.freq_tracker import (
    FreqTracker,
)

from sae_components.components.resampling.ema import EMAFreqTracker


class FreqTracked(cl.Module):
    def __init__(self, module: cl.CacheLayer, freq_tracker: cl.Module = None):
        super().__init__()
        self.module = module
        self.freq_tracker: FreqTracker = freq_tracker or EMAFreqTracker()

    def forward(self, x, *, cache: cl.Cache = None, **kwargs):
        acts = self.module(*x, cache=cache, **kwargs)
        self.freq_tracker.update(acts)
        return acts

    @property
    def freqs(self):
        return self.freq_tracker.freqs

    @property
    def weight(self):
        return self.module.weight

    @property
    def bias(self):
        return self.module.bias
