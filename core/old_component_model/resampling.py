from nqgl.mlutils.components.config import WandbDynamicConfig
from nqgl.mlutils.components.old_component_model.freq_component import (
    FreqMonitorComponent,
)
from nqgl.mlutils.components.old_component_model.act_freq_layer import ActFreqLayer
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ResamplingConfig(WandbDynamicConfig):
    dead_threshold: float = 3e-6
    min_viable_count: int = 10_000
    reset_to_freq: float = 0.001
    reset_to_count: int = 10_000
    check_frequency: int = 100


class ResamplingLayer(torch.nn.Module):
    def __init__(
        self,
        cfg: ResamplingConfig,
        freq_layer: ActFreqLayer,
        downstream_weights: nn.Parameter,
    ):
        self.freq_layer = freq_layer
        self.downstream_weights = downstream_weights
        self.T = 0

    def get_dead_neurons(self): ...

    # def re_init_neurons
    # -> rename "resample"

    # def resampling_check

    # def reset_neurons

    # def reset_activation_frequencies
    #     -> reset_freqs

    # def get_activation_frequencies(self):
    #     -> freqs property


class QueuedResettingLinear:
    re_init_queued: callable
    ...

    @torch.no_grad()
    def re_init_neurons(self, x_diff): ...


class GhostGradResettingLinear: ...


# class
