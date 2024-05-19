import torch
from typing import List
from sae_components.components.resampling.freq_tracker import FreqTracker
import sae_components.core as cl


class EMAFreqTracker(FreqTracker):
    def __init__(self, beta=0.99):
        self.activation_freqs = None
        self.beta = beta

    @torch.no_grad()
    def update(self, acts):
        freqs = (acts > 0).float().mean(dim=0)
        if self.activation_freqs is None:
            self.activation_freqs = torch.zeros_like(freqs).float() + 1e-5
        self.activation_freqs.lerp_(freqs, 1 - self.beta)

    @property
    def freqs(self):
        return self.activation_freqs

    def reset(self):
        self.activation_freqs = torch.zeros_like(self.activation_freqs) + 1e-5
