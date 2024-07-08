import torch
from .freq_tracker import FreqTracker


class EMAFreqTracker(FreqTracker):
    def __init__(self, beta=0.995):
        super().__init__()
        self.activation_freqs = None
        self.beta = beta

    @torch.no_grad()
    def process_data(self, acts, cache, **kwargs):
        freqs = (acts > 0).float().mean(dim=0)
        if self.activation_freqs is None:
            self.activation_freqs = torch.zeros_like(freqs).float() + 1e-5
        self.activation_freqs.lerp_(freqs, 1 - self.beta)

    @property
    def freqs(self):
        return self.activation_freqs

    def reset(self):
        self.activation_freqs = torch.zeros_like(self.activation_freqs) + 1e-5
