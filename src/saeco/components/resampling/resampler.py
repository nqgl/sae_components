from typing import Optional
import torch
import torch.nn as nn

from saeco.components.features.features_param import FeaturesParam
from saeco.components.features.optim_reset import (
    OptimResetValues,
    OptimResetValuesConfig,
)
from .freq_tracker import FreqTracker
from saeco.components.features import (
    LinDecoder,
    LinEncoder,
    EncoderBias,
    Resamplable,
    ResampledWeight,
    HasFeatures,
)


def get_resampled_params(model: nn.Module) -> set[FeaturesParam]:
    l: set[FeaturesParam] = set()
    for m in model.modules():
        if isinstance(m, HasFeatures):
            l |= set(m.features.values())
    d = {}
    for fp in l:
        if fp.param in d:
            other = d[fp.param]
            assert other == fp, f"{other} != {fp}"
            raise ValueError(
                f"Duplicate feature parameter {fp}. implement __eq__ and change this check to just (intelligently) deduplicate and check for inconsistency"
            )
        d[fp.param] = fp
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
from saeco.sweeps import SweepableConfig
from pydantic import Field


class ResamplerConfig(SweepableConfig):
    optim_reset_cfg: OptimResetValuesConfig = Field(
        default_factory=OptimResetValuesConfig
    )
    bias_reset_value: float = 0
    dead_threshold: float = 3e-6
    freq_balance: Optional[int | float] = None
    freq_balance_strat: str = "sep"
    expected_biases: Optional[int] = 1
    expected_decs: Optional[int] = 1
    expected_encs: Optional[int] = 1


class Resampler(ABC):
    """
    currently it is assumed that the freq tracker does not change after
    the first usage
    """

    def __init__(
        self,
        cfg: ResamplerConfig,
    ):
        self.cfg = cfg
        self.model = None
        self._encs = None
        self._decs = None
        self._biases = None
        # self._freq_tracker = None

    def get_feature_indices_to_reset(self):
        return self.freq_tracker.freqs < self.cfg.dead_threshold

    def get_reset_feature_directions(self, num_directions, data_source, model): ...

    def resample(self, data_source: iter, optimizer: torch.optim.Optimizer, model):
        i = self.get_feature_indices_to_reset()
        d = self.get_reset_feature_directions(
            num_directions=sum(i) if i.dtype is torch.bool else len(i),
            data_source=data_source,
            model=model,
        )
        assert self.cfg.expected_encs is None or self.cfg.expected_encs == len(
            self.encs
        )
        assert self.cfg.expected_biases is None or self.cfg.expected_biases == len(
            self.biases
        )
        assert self.cfg.expected_decs is None or self.cfg.expected_decs == len(
            self.decs
        )
        for r in self.encs + self.decs + self.biases:
            r.set_cfg(self.cfg)
            r.resample(
                indices=i,
                new_directions=d,
                bias_reset_value=self.cfg.bias_reset_value,
                optim=optimizer,
            )
        if self.cfg.freq_balance is not None:
            self.wholistic_freqbalance(
                model=model,
                datasrc=data_source,
                indices=i,
                target_l0=self.cfg.freq_balance,
                target_l1=None,  # self.cfg.freq_balance,
            )

    def assign_model(self, model):
        self.model = model
        return model

    def setup_resample_types(self):
        encs = []
        decs = []
        biases = []
        for param in get_resampled_params(self.model):
            if param.type == "enc":
                encs.append(param)
            elif param.type == "dec":
                decs.append(param)
            elif param.type == "bias":
                biases.append(param)
            else:
                raise ValueError(
                    f'Unexpected resample site {param} of type {param.type}"'
                )

        if self._encs is None:
            self._encs = encs
        if self._decs is None:
            self._decs = decs
        if self._biases is None:
            self._biases = biases

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

    @torch.no_grad()
    def bias_freqbalance(
        self,
        model,
        datasrc,
        indices,
        target_l0,
    ):

        original_beta = self.freq_tracker.beta
        self.freq_tracker.beta = 0.9
        for bias in self.biases:
            bias.param.data[indices] = -1
        for i in range(70):
            lr = 3000 / (1.01**i)
            if i < 10:
                lr /= 2 ** (10 - i)
            for _ in range(10):
                with torch.autocast("cuda"):
                    d = next(datasrc)
                    model(d)
                target_freq = target_l0 / self.freq_tracker.freqs.shape[0]
                fn = lambda x: x
                freqs = self.freq_tracker.freqs[indices]
                if self.cfg.freq_balance_strat == "mean":
                    freqs = freqs.mean() * 0.99 + freqs * 0.01
                elif self.cfg.freq_balance_strat == "mix":
                    freqs = freqs.mean() / 2 + freqs / 2
                diff = fn(torch.tensor(target_freq)) - fn(freqs)
                for bias in self.biases:
                    # b2z = (
                    #     bias.param.data[indices]
                    #     + torch.relu(bias.param.data[indices]) * 2
                    # )
                    # bdiff = diff - b2z * 0.001
                    bias.param.data[indices] *= 1.3 ** (-diff)
            print(
                i,
                diff.abs().mean().item(),
                self.freq_tracker.freqs[indices].mean()
                * self.freq_tracker.freqs.shape[0],
            )
            # self.freq_tracker.reset()
        self.freq_tracker.beta = original_beta

    @torch.no_grad()
    def wholistic_freqbalance(
        self,
        model,
        datasrc,
        indices=None,
        target_l0=45,
        target_l1=None,
    ):
        from saeco.trainer.train_cache import TrainCache

        # target_l1 = target_l1 or target_l0
        if indices is None:
            indices = torch.ones(self.encs[0].features.shape[0], dtype=torch.bool)

        # self.bias_freqbalance(
        #     model=model, datasrc=datasrc, indices=indices, target_l0=target_l0
        # )
        ### DO BIAS FREQ BALANCING:
        original_beta = self.freq_tracker.beta
        datas = []
        self.freq_tracker.beta = 0.8
        if target_l0 is not None:
            for bias in self.biases:
                bias.param.data[indices] = -1
        for i in range(200):
            lr = 5 / (1.017**i)
            if i < 10:
                lr /= 2 ** (10 - i)
            for _ in range(3):
                with torch.autocast("cuda"):
                    d = next(datasrc)
                    datas.append(d)
                    if target_l0 is None:
                        continue
                    model(d)
                target_freq = target_l0 / self.freq_tracker.freqs.shape[0]
                fn = lambda x: x
                freqs = self.freq_tracker.freqs[indices]
                if self.cfg.freq_balance_strat == "mean":
                    freqs = freqs.mean() * 0.99 + freqs * 0.01
                elif self.cfg.freq_balance_strat == "mix":
                    freqs = freqs.mean() / 2 + freqs / 2
                diff = fn(torch.tensor(target_freq)) - fn(freqs)
                for bias in self.biases:
                    # b2z = (
                    #     bias.param.data[indices]
                    #     + torch.relu(bias.param.data[indices]) * 2
                    # )
                    # bdiff = diff - b2z * 0.001
                    bias.param.data[indices] *= 2 ** (-diff * lr)
            if target_l0 is None:
                continue

            print(
                i,
                diff.abs().mean().item(),
                self.freq_tracker.freqs[indices].mean()
                * self.freq_tracker.freqs.shape[0],
            )
            # self.freq_tracker.reset()
        self.freq_tracker.beta = original_beta
        if target_l1 is None:
            return
        num_reset = (
            indices.sum().item() if indices.dtype is torch.bool else len(indices)
        )
        ### DO MAGNITUDE STEP
        acts_acc = torch.zeros(num_reset).cuda()
        for d in datas:
            cache = TrainCache()
            cache.acts = ...
            with torch.autocast("cuda"):
                d = next(datasrc)
                model(d, cache=cache)
            acts_acc += cache.acts.relu().mean(0)[indices]
        acts_avg = acts_acc / len(datas)
        target_act = target_l1 / self.biases[0].features.shape[0]
        scale = torch.where(acts_avg > 1e-6, target_act / acts_avg, 1e3)
        for enc in self.encs:
            enc.features[indices] = enc.features[indices] * scale.unsqueeze(-1)
        for bias in self.biases:
            bias.features[indices] = bias.features[indices] * scale
