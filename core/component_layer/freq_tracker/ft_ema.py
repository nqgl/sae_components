from nqgl.mlutils.components.cache_layer import ActsCache
from nqgl.mlutils.components.component_layer.freq_tracker.freq_tracker import (
    CountingFreqTrackerConfig,
    FreqTracker,
)
from nqgl.mlutils.components.config import WandbDynamicConfig
import torch

from dataclasses import dataclass


@dataclass
class EMAFreqTrackerConfig(WandbDynamicConfig):
    reset_to_freq: float = 1 / 768
    initial_freq_value: float = 1e-5
    reset_ema_to_freq: float = 1 / 768

    # reset_to_count: int = 0
    decay: float = 0.9992


class EMAFreqTracker(FreqTracker):
    train_cache_watch = ["acts"]
    eval_cache_watch = []
    cfg: EMAFreqTrackerConfig

    def __init__(self, cfg=EMAFreqTrackerConfig()):
        super().__init__()
        self.cfg = cfg
        self.emactfreqs = self.cfg.initial_freq_value
        # self._parent = None
        # self.parent: CacheProcLayer = parent

    def get_dead_neurons(self, count_min, threshold):
        return self.freqs < threshold  # & (self._count > count_min)

    def _update_from_cache(self, cache: ActsCache, **kwargs):
        if cache.has.acts:
            self.emactfreqs = self.emactfreqs * self.cfg.decay + self.emactfreq_calc(
                cache
            ) * (1 - self.cfg.decay)

    def emactfreq_calc(self, cache: ActsCache):
        return cache.acts.count_nonzero(dim=0).float() / cache.acts.shape[0]

    def reset_freqs(self, mask=None, initial_activation=0, initial_count=0):
        if mask is None:
            # assert isinstance(self._count, int)
            # self.emactfreqs[mask] = (
            #     torch.zeros_like(self.emactfreqs) + self.cfg.reset_to_freq
            # )
            pass
        else:
            self.emactfreqs[mask] = self.cfg.reset_ema_to_freq
            # self._count[mask] = initial_count

    @property
    def freqs(self):
        return self.emactfreqs
