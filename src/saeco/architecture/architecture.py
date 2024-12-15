from typing import Callable, TypeVar, Generic
from pydantic import BaseModel
from attrs import define, field
from saeco.components.metrics.metrics import ActMetrics, PreActMetrics
from saeco.components.resampling.freq_tracker.ema import EMAFreqTracker
from saeco.core.cache import Cache
from saeco.misc.utils import useif

# Assuming these classes are defined or imported from your codebase
from saeco.trainer import TrainingRunner
from saeco.core import Module
from saeco.components.losses import Loss
from saeco.initializer.initializer import Initializer
from saeco.sweeps import SweepableConfig
from saeco.trainer.normalizers.normalizer import (
    StaticInvertibleGeneralizedNormalizer,
    NormalizedIO,
    NormalizedInputs,
    DeNormalizedOutputs,
)
from saeco.trainer.run_config import RunConfig
import saeco.core as cl
import saeco.components as co
from functools import cached_property
from torch import nn
from abc import abstractmethod

ArchConfigType = TypeVar("ArchConfigType", bound=SweepableConfig)


# class Model(Generic[ArchConfigType], cl.Module):
#     normalizer: StaticInvertibleGeneralizedNormalizer
#     _encoder_pre: cl.Module = NotImplemented
#     nonlinearity: cl.Module = NotImplemented
#     _decoder: cl.Module = NotImplemented
#     encoder: cl.Module = NotImplemented
#     pre_acts: cl.Module = NotImplemented
#     losses: list[Loss] = field(factory=list)

#     def encode(self, x, cache: cl.Cache, **kwargs): ...

#     def decode(self, x, cache: cl.Cache, **kwargs): ...

#     def encode_pre(self, x, cache: cl.Cache, **kwargs): ...

#     def forward(self, x, *, cache: cl.Cache, **kwargs):
#         x = cache(self).encode(x)
#         x = cache(self).decode(x)
#         return x


class SAE(Generic[ArchConfigType], cl.Seq):
    encoder_pre: cl.Module
    preacts: PreActMetrics
    nonlinearity: cl.Module
    encoder: cl.Module
    decoder: cl.Module
    acts: ActMetrics = None
    freqs: EMAFreqTracker | None
    penalty: co.Penalty | None
    losses: list[Loss]

    def __init__(
        self,
        encoder_pre: cl.Module | None = None,
        nonlinearity: cl.Module | None = None,
        decoder: cl.Module = None,
        encoder: cl.Module | None = None,
        acts: ActMetrics = None,  # rename to "metrics"?
        # when its added to the architecture, if it's an aux model rename the acts
        preacts: PreActMetrics = None,
        penalty: co.Penalty | None = ...,
        freqs: EMAFreqTracker | None = ...,
    ):
        penalty = co.L1Penalty() if penalty is ... else penalty
        freqs = EMAFreqTracker() if freqs is ... else freqs
        acts = ActMetrics() if acts is None else acts
        preacts = PreActMetrics() if preacts is None else preacts
        # we could seek a preactmetrics on the encoder in the future
        assert (encoder_pre is None) == (nonlinearity is None)
        assert (encoder is None) != (nonlinearity is None)
        if encoder is None:
            encoder = cl.Seq(
                encoder_pre=encoder_pre,
                preacts=preacts,
                nonlinearity=nonlinearity,
            )
        self.losses = nn.ModuleList()
        super().__init__(
            # encoder_pre=encoder_pre,
            # preacts=preacts,
            # nonlinearity=nonlinearity,
            encoder=encoder,
            acts=acts,
            freqs=freqs,
            **useif(penalty is not None, penalty=penalty),
            decoder=decoder,
        )

    def __lshift__(self, other):
        if issubclass(other, Loss):
            return self.losses.append(other(self))
        elif isinstance(other, Loss):
            assert (
                other.module.module is self
            ), "if this causes issues, we may want to seek self instead or add flag for complex architectures"
            return self.losses.append(other)

    # @cached_property
    # def encode(self):
    #     return self[:"acts"]

    # @cached_property
    # def decode(self):
    #     return self.decoder

    # # encode: cl.Module = NotImplemented
    # @classmethod
    # def build_model(
    #     cls,
    #     normalizer: StaticInvertibleGeneralizedNormalizer,
    #     encoder_pre: cl.Module,
    #     nonlinearity: cl.Module,
    #     decoder: cl.Module,
    # ):
    #     return cls(
    #         encoder_pre=encoder_pre,
    #         nonlinearity=nonlinearity,
    #         decoder=decoder,
    #     )


# class NormalizedModel(cl.Module):
#     def __init__(self, normalizer, model: SAE) -> None:
#         super().__init__()
#         self.normalizer = normalizer


# @define
class Architecture(Generic[ArchConfigType], cl.Module):
    """
    multiple ways to set up an architecture

    override constructor methods (make_*)
    set the modules fields
    override the functional methods (encode, decode)

    also for encoder in particular, it can either be defined by:
    - encoder_pre + nonlinearity
    - encode

    """

    normalizer: StaticInvertibleGeneralizedNormalizer
    primary_model: SAE | None = field(default=None)
    aux_models: list[SAE] | None = field(default=None)

    def __init__(self, run_cfg: RunConfig[ArchConfigType]):
        self.run_cfg = run_cfg
        self.cfg = None
        self.instantiated = False
        self._aux_models = nn.ModuleList()
        self._model = None
        self._losses = nn.ModuleList()

    def instantiate(self, inst_cfg=None):
        if inst_cfg:
            self.run_cfg = self.run_cfg.from_selective_sweep(inst_cfg)
        self.cfg = self.run_cfg.arch_cfg
        assert self.cfg.is_concrete() and self.run_cfg.is_concrete()
        self.instantiated = True
        self.setup()

    @abstractmethod
    def setup(self): ...

    def build(self):
        raise NotImplementedError

    # def __attrs_post_init__(self):
    #     if self.__class__ == Architecture:
    #         raise Exception(
    #             "Architecture class should be subclassed before instantiation"
    #         )

    # def encode_pre(self, x, cache: cl.Cache, **kwargs): ...

    # def encode(self, x, *, cache: cl.Cache, **kwargs):
    #     return cache(self).encoder(x)  # TODO normalization

    # def decode(self, x, *, cache: cl.Cache, **kwargs):
    #     return cache(self).decoder(x)  # TODO normalization

    # def forward(self, x, *, cache: cl.Cache, **kwargs):
    #     x = cache(self).encode(x)
    #     x = cache(self).decode(x)
    #     return x

    @cached_property
    def init(self) -> Initializer:
        return Initializer(
            self.run_cfg.init_cfg.d_data,
            dict_mult=self.run_cfg.init_cfg.dict_mult,
            l0_target=self.run_cfg.train_cfg.l0_target,
        )

    @cached_property
    def model(self) -> SAE:
        return SAE(
            encoder_pre=self.make_encoder_pre(),
            nonlinearity=self.make_nonlinearity(),
            encoder=self.make_encoder(),
            decoder=self.make_decoder(),
        )

    def run(cfg):

        tr = TrainingRunner(cfg, model_fn=anth_update_model)
        tr.trainer.train()
