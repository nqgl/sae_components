import torch
from jaxtyping import Float, jaxtyped, Bool
from torch import Tensor
from nqgl.mlutils.components.cache_layer import CacheLayer, CacheProcLayer
from typing import Any, Union, Optional
from dataclasses import field

from nqgl.mlutils.components.old_component_model.freq_component import (
    FreqMonitorComponent,
)


class ActFreqLayer(CacheProcLayer):
    def __init__(
        self, cfg, cachelayer: CacheLayer, freq_monitor_cls: FreqMonitorComponent
    ):
        super().__init__(cachelayer)
        self.cfg = cfg
        self.activations: FreqMonitorComponent = freq_monitor_cls(self)
