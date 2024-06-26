import torch
import torch.nn as nn
import saeco.core as cl
from saeco.components.sae_cache import SAECache
from torch import Tensor
from jaxtyping import Float

from typing import Tuple, Any, Union, Optional, List
from saeco.components.resampling.freq_tracked import (
    FreqTracked,
)


class ResampledLayer(cl.Seq):
    layer: FreqTracked

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
