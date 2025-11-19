import torch
from .freq_tracker import FreqTracker


class EMAFreqTracker(FreqTracker):
    def __init__(self, beta=0.99, encoder_index: int | None = 0):
        super().__init__(encoder_index=encoder_index)
        self.activation_freqs = None
        self.beta = beta

    def update_freqs(self, freqs, cache):
        if self.activation_freqs is None:
            self.activation_freqs = torch.zeros_like(freqs).float() + 1e-5
        self.activation_freqs.lerp_(freqs, 1 - self.beta)

    @property
    def freqs(self):
        return self.activation_freqs

    def reset(self):
        self.activation_freqs = torch.zeros_like(self.activation_freqs) + 1e-5
