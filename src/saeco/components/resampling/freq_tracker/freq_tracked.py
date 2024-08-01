import saeco.core as cl

from saeco.components.resampling.freq_tracker.freq_tracker import (
    FreqTracker,
)

from saeco.components.resampling.freq_tracker.ema import EMAFreqTracker
import saeco.core.module


class FreqTracked(saeco.core.module.Module):
    def __init__(
        self,
        module: cl.CacheLayer,
        freq_tracker: saeco.core.module.Module = None,
    ):
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
