from typing import Optional
import torch
import torch.nn as nn
from .freq_tracker import FreqTracker
from saeco.components.features import (
    LinDecoder,
    LinEncoder,
    EncoderBias,
    Resamplable,
    ResampledWeight,
)

# from saeco.trainer.trainable import Trainable # TODO circular, maybe just define a protocol for the model passed to resampler


def get_resample_sites(model: nn.Module) -> set[Resamplable]:
    l = set()
    for m in model.modules():
        if isinstance(m, Resamplable):
            l.add(m)
    return l


def get_freq_trackers(model: nn.Module):
    l = set()
    for m in model.modules():
        if isinstance(m, FreqTracker):
            l.add(m)
    return l


def find_matching_submodules(module: nn.Module, matchfn):
    matches = set()

    def appl_fn(m):
        if matchfn(m):
            matches.add(m)

    module.apply(appl_fn)
    return matches


from abc import ABC
from saeco.misc import lazycall


class Resampler(ABC):
    """
    currently it is assumed that the freq tracker does not change after
    the first usage
    """

    def __init__(
        self,
        expected_biases: Optional[int] = 1,
        expected_decs: Optional[int] = 1,
        expected_encs: Optional[int] = 1,
    ):
        self.expected_biases = expected_biases
        self.expected_decs = expected_decs
        self.expected_encs = expected_encs
        self.model = None
        self._encs = None
        self._decs = None
        self._biases = None
        # self._freq_tracker = None

        self.bias_reset_value = -0.3
        self.dead_threshold = 3e-6

    def get_feature_indices_to_reset(self):
        return self.freq_tracker.freqs < self.dead_threshold

    def get_reset_feature_directions(self, num_directions, data_source, model): ...

    def resample(self, data_source: iter, optimizer: torch.optim.Optimizer, model):
        i = self.get_feature_indices_to_reset()
        d = self.get_reset_feature_directions(
            num_directions=sum(i) if i.dtype is torch.bool else len(i),
            data_source=data_source,
            model=model,
        )
        assert self.expected_encs == len(self.encs)
        assert self.expected_biases == self.n_biases
        assert self.expected_decs == len(self.decs)
        for r in self.encs + self.decs + self.biases:
            r.resample(
                indices=i, new_directions=d, bias_reset_value=self.bias_reset_value
            )

    def assign_model(self, model):
        self.model = model
        return model

    def setup_resample_types(self):
        encs = []
        decs = []
        biases = []

        for r in get_resample_sites(self.model):  # TODO
            if isinstance(r, ResampledWeight) and isinstance(r.wrapped, LinEncoder):
                encs.append(r)
            elif isinstance(r, ResampledWeight) and isinstance(r.wrapped, LinDecoder):
                decs.append(r)
            elif isinstance(r, EncoderBias):
                biases.append(r)
            else:
                raise ValueError(
                    f"Unexpected resample site {r} of type {type(r)}\
                    may just need to add support for matmul types here"
                )

        if self._encs is None:
            self._encs = encs
        if self._decs is None:
            self._decs = decs
        if self._biases is None:
            self._biases = biases

    @property
    def n_biases(self):
        return len(self.biases) + len([e for e in self.encs if e.bias is not None])

    @property
    @lazycall
    def freq_tracker(self) -> FreqTracker:
        fts = get_freq_trackers(self.model)
        assert len(fts) == 1, f"Expected 1 freq tracker, got {len(fts)}"
        ft: FreqTracker = fts.pop()
        return ft

    def set_freq_tracker(self, value):
        assert self._freq_tracker is None
        self._freq_tracker = value
        return value

    @property
    def encs(self) -> list[Resamplable]:
        if self._encs is None:
            self.setup_resample_types()
        return self._encs

    def add_to_encs(self, enc):
        if self._encs is None:
            self._encs = []
        self._encs.append(enc)
        return enc

    @property
    def decs(self) -> list[Resamplable]:
        if self._decs is None:
            self.setup_resample_types()
        return self._decs

    def add_to_decs(self, dec):
        if self._encs is None:
            self._encs = []
        self._encs.append(dec)
        return dec

    @property
    def biases(self) -> list[Resamplable]:
        if self._biases is None:
            self.setup_resample_types()
        return self._biases

    def add_to_biases(self, bias):
        if self._encs is None:
            self._encs = []
        self._encs.append(bias)
        return bias
