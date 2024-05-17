import torch
import torch.nn as nn
import sae_components.core as cl
from sae_components.components.sae_cache import SAECache
from torch import Tensor
from jaxtyping import Float

from typing import Tuple, Any, Union, Optional, List
from sae_components.components.resampling.freq_tracker.ema import (
    EMAFreqTracker,
    FreqTracker,
)


class FreqTracked(cl.Sequential):
    def __init__(self, module: cl.CacheLayer, freq_tracker: cl.CacheLayer = None):
        super().__init__()
        freq_tracker = freq_tracker or EMAFreqTracker(module.out_features)
        self.module = module
        self.freq_tracker: FreqTracker = freq_tracker

    def forward(self, *x, cache: cl.Cache = None, **kwargs):
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
