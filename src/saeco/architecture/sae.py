from typing import Literal, overload

from torch import nn

import saeco.components as co
import saeco.core as cl
from saeco.components.losses import Loss
from saeco.components.metrics.metrics import ActMetrics, PreActMetrics
from saeco.components.resampling.freq_tracker.ema import EMAFreqTracker
from saeco.misc.utils import useif


class SAE(cl.Seq):
    """The core sparse autoencoder module: a configurable encode → nonlinearity
    → decode sequence.

    A composed ``Seq`` of named stages (``encoder_pre``, ``nonlinearity``,
    ``encoder``, ``decoder``, plus activation metrics, an optional
    frequency tracker, and an optional sparsity penalty). Construct it
    inside an ``Architecture``'s ``@model_prop`` from the building blocks
    in ``saeco.core`` / ``saeco.components`` and the parameter factories
    on ``self.init``.
    """

    encoder_pre: cl.Module
    preacts: PreActMetrics
    nonlinearity: cl.Module
    encoder: cl.Module
    decoder: cl.Module
    acts: ActMetrics
    freqs: EMAFreqTracker | None
    penalty: co.Penalty | None
    losses: list[Loss]

    @overload
    def __init__(
        self,
        *,
        encoder_pre: Literal[None] = None,
        nonlinearity: Literal[None] = None,
        encoder: nn.Module,
        decoder: nn.Module,
        act_metrics: ActMetrics | None = None,
        preacts: PreActMetrics | None = None,
        penalty: co.Penalty | None = ...,
        freqs: EMAFreqTracker | None = ...,
    ): ...

    @overload
    def __init__(
        self,
        *,
        encoder_pre: nn.Module,
        nonlinearity: nn.Module,
        encoder: Literal[None] = None,
        decoder: nn.Module,
        act_metrics: ActMetrics | None = None,
        preacts: PreActMetrics | None = None,
        penalty: co.Penalty | None = ...,
        freqs: EMAFreqTracker | None = ...,
    ): ...
    def __init__(
        self,
        *,
        encoder_pre: nn.Module | None = None,
        nonlinearity: nn.Module | None = None,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        act_metrics: ActMetrics | None = None,
        preacts: PreActMetrics | None = None,
        penalty: co.Penalty | None = ...,
        freqs: EMAFreqTracker | None = ...,
    ):
        penalty = co.L1Penalty() if penalty is ... else penalty
        freqs = EMAFreqTracker() if freqs is ... else freqs
        act_metrics = ActMetrics() if act_metrics is None else act_metrics
        # we could seek a preactmetrics on the encoder in the future
        assert (encoder_pre is None) == (nonlinearity is None)
        assert (encoder is None) != (nonlinearity is None)
        if encoder is None:
            preacts = PreActMetrics() if preacts is None else preacts
            encoder = cl.Seq(
                encoder_pre=encoder_pre,
                preacts=preacts,
                nonlinearity=nonlinearity,
            )
        else:
            assert preacts is None
        super().__init__(
            # normalizer=normalizer,
            # encoder_pre=encoder_pre,
            # preacts=preacts,
            # nonlinearity=nonlinearity,
            encoder=encoder,
            acts=act_metrics,
            **useif(freqs is not None, freqs=freqs),
            **useif(penalty is not None, penalty=penalty),
            decoder=decoder,
            # denormalizer=...
        )

    def set_to_aux_model(self, aux_name: str):
        self.acts.name = aux_name
