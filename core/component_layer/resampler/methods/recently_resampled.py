import torch.nn as nn
from torch import Tensor
from nqgl.mlutils.components.component_layer.resampler.resampler import (
    ResamplerConfig,
    ResamplingMethod,
)
from typing import Optional

from dataclasses import dataclass


@dataclass
class RecentlyResampledTrackerConfig(ResamplerConfig):
    save_last_n_resamples: int = 10  # no this should be time based


class RecentlyResampledTracker(ResamplingMethod):
    def __init__(
        self, cfg: ResamplerConfig, W_next: Optional[nn.Parameter] = None, **kwargs
    ):
        super().__init__(cfg=cfg, W_next=W_next, **kwargs)
        self.recently_resampled = None

    def reset_neurons(self, new_directions: Tensor, to_reset: Tensor):
        return super().reset_neurons(new_directions, to_reset)
