import torch.nn as nn
from nqgl.mlutils.components.component_layer.resampler.resampler import (
    ResamplerConfig,
    ResamplingCache,
    ResamplingMethod,
)


from dataclasses import dataclass
from typing import Tuple, Optional
import torch


@dataclass
class QueuedResamplerConfig(ResamplerConfig):
    resample_frequency: int = 100
    resampling_cycle: Tuple[int, int] = 1, 1
    append_to_queue: bool = True


class QueuedResampler(ResamplingMethod):
    def __init__(
        self,
        cfg: QueuedResamplerConfig,
        W_next: Optional[nn.Parameter] = None,
        **kwargs,
    ):
        super().__init__(cfg=cfg, W_next=W_next, **kwargs)
        self.queued = None

    def is_resample_step(self):
        return (
            self.T % self.cfg.resample_frequency == 0
            and self._layer.training
            and self.queued is not None
            and self.queued.shape[0] > 0
        )

    def get_dead_neurons_for_norm(self):
        if self.queued is None:
            return self.get_dead_neurons()
        mask = self.get_dead_neurons()
        mask[:] = False
        mask[self.queued] = True
        return mask

    def check_dead(self):
        super().check_dead()
        self.queued = (
            torch.unique(
                torch.cat((self.queued, self.get_dead_neurons().nonzero())),
                sorted=False,
                dim=0,
            )
            if self.queued is not None and self.cfg.append_to_queue
            else self.get_dead_neurons().nonzero()
        )

    def _update_from_cache(self, cache: ResamplingCache, **kwargs):
        cache.num_queued_for_reset = (
            self.queued.shape[0] if self.queued is not None else 0
        )
        return super()._update_from_cache(cache, **kwargs)

    def get_neurons_to_resample(self):
        if (
            self.queued is not None
            and self.T % self.cfg.resampling_cycle[1]
            > self.cfg.resampling_cycle[1] - self.cfg.resampling_cycle[0]
        ):
            q = self.queued[: self.cfg.num_to_resample]
            self.queued = self.queued[self.cfg.num_to_resample :]
            return q
        return None
