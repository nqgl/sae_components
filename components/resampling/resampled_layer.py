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


class ResampledLayer(cl.Sequential):
    layer: cl.CacheLayer
    freq_tracker: FreqTracker

    def __init__(self, layer, freq_tracker, resampler):
        super().__init__(
            layer=layer,
            freq_tracker=freq_tracker,
        )
        self.resampler = resampler


class FreqTracked(cl.Sequential):
    layer: cl.CacheLayer
    freq_tracker: FreqTracker

    def __init__(self, layer, freq_tracker, resampler):
        super().__init__(
            layer=layer,
            freq_tracker=freq_tracker,
        )
        self.resampler = resampler


class Resampler:  # this is NOT a module
    def __init__(self, encoder_layer, decoder_layer: cl.CacheLayer):
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
