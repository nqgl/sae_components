"""Graph-native analogs of saeco's ``SAE`` / ``Loss`` / ``Normalizer`` /
``Trainable`` -- everything is a ``GraphSpec`` you compose by plugging one graph
into another's ``sub_slot`` and applying it (``model(x).acts``).

The normalization win over stock saeco: instead of the awkward
``normalizer[loss[model]]``, :class:`NormalizedModel` is a graph that wraps a
model graph (``x -> normalize -> model -> denormalize``), and a loss is a graph
that plugs the normalized model in and reads its normalized handles -- the
natural ``loss[normalizer[model]]``. The normalize node is shared, so it all
fuses to one encoder/normalizer pass.

Assembly mirrors stock saeco: ``@model_prop`` returns a model graph (an
:class:`SAEGraphSpec`), ``@loss_prop`` returns ``GraphL2Loss.on(model)`` (a
:class:`LossRef`). :class:`GraphTrainable` wraps each model in a
:class:`NormalizedModel` and resolves every loss against it.
"""

from functools import reduce
from typing import Protocol

import torch
from attrs import define

import saeco.core as cl
from saeco.data.training_data.sae_train_batch import SAETrainBatch
from saeco.trainer.trainable import Trainable, make_cache_optional
from saeco_research.cgraph2.core import (
    Graph,
    Input,
    Op,
    OpRef,
    ValueRef,
    bind,
    collect,
    op,
    ref,
)
from saeco_research.cgraph2.spec import GraphSpec, arg, op_slot, sub_slot


# --- graph-leaf math (module-level so consing keys on function structure) -----
def _mse(a, b):
    return ((a - b) ** 2).mean()


def _l0(a):
    return (a != 0).sum(-1).float().mean(0).sum()


def _l1(a):
    return a.abs().mean(0).sum()


def _cosim(a, b):
    return torch.cosine_similarity(a, b, dim=-1).mean()


# --- bubble-up sites: passthrough markers on `acts` that expose extra handles --
# A site is collected from a graph cone (apply-style, `collect()`); the outside
# pulls its handles without anything in between knowing where it lives -- the
# graph-native version of saeco's ActMetrics "bubbling up" to the cache (D-03/D-05).
class ActMetrics(GraphSpec):
    """Metrics on activations, as a graph: ``acts`` in, ``L0`` / ``L1`` out (and
    the passthrough ``output``). Applied to whatever acts flow through its site,
    so it tracks rebinding (e.g. normalization)."""

    acts: ValueRef = arg()
    L0 = op(_l0)(acts)
    L1 = op(_l1)(acts)
    output = acts


@define(eq=False)
class _MetricsSite:
    """A passthrough marker spliced onto ``acts`` (``metered(acts)``). Collected by
    ``collect(_MetricsSite)``; ``site.at(acts)`` applies the carried metrics graph
    to the (already-rebound) acts flowing through it."""

    metrics: type = ActMetrics

    def __call__(self, x):
        return x

    def at(self, acts: ValueRef) -> "GraphSpec":
        return self.metrics(acts=acts)


def metered(acts: ValueRef, metrics: type = ActMetrics) -> ValueRef:
    """Splice a metrics site onto ``acts`` (passthrough); bubble it up with
    ``collect(_MetricsSite)``."""
    return ref(_MetricsSite(metrics), (acts,))


class FreqTracker(cl.Module):
    """A passthrough freq site on ``acts``: tracks an EMA of per-feature firing
    rate. ``updated(acts)`` is the handle for the new freqs given this step's
    acts; ``update(new)`` writes them back. Called with a handle inside a
    ``forward_values`` block, ``update`` *defers* the write to block exit -- so
    you can write ``ft.update(ft.updated(acts))`` inline and it runs after the
    pass (the principled before/after-update split: ``freqs`` and ``updated_freqs``
    are distinct nodes; only an explicit update advances the state)."""

    def __init__(self, d_dict: int, decay: float = 0.99):
        super().__init__()
        self.decay = decay
        self.register_buffer("freqs", torch.zeros(d_dict))

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return x  # passthrough

    def updated(self, acts: ValueRef) -> ValueRef:
        ema = OpRef(Op.create(self._ema, "freq_ema", key=("freq_ema", id(self))))
        return ema(ref(self.freqs, ()), acts)

    def _ema(self, freqs, acts):
        firing = (acts != 0).float().mean(0)
        return freqs * self.decay + firing * (1 - self.decay)

    @torch.no_grad()
    def update(self, new_freqs) -> None:
        from saeco_research.cgraph2.core import _ACTIVE_SESSION

        session = _ACTIVE_SESSION.get()
        if isinstance(new_freqs, ValueRef) and session is not None:
            session.after(new_freqs, self.update)  # defer to the block's exit
            return
        self.freqs.copy_(new_freqs)


class SAEModel(Protocol):
    """Typing protocol for a model graph: ``acts`` / ``recon`` handles, and
    appliable to an input (``model(x)``). Any :class:`SAEGraphSpec` satisfies it."""

    acts: ValueRef
    recon: ValueRef

    def __call__(self, x: ValueRef) -> "SAEModel": ...


class SAEGraphSpec(GraphSpec):
    """Base for SAE model graphs. Subclasses define ``acts`` / ``recon`` /
    ``output`` handles; this adds the model-level hook the framework needs."""

    name = None

    def set_to_aux_model(self, name: str):
        self.name = name
        return self


class AffineNormalizer(cl.Module):
    """A clean, static, invertible affine normalizer: center by a scalar mean,
    scale by a per-feature std times a scalar std (both primed from data).

    This is the affine map saeco's ``StaticInvertibleGeneralizedNormalizer``
    computes under its default config, minus the GNConfig/aggregation-enum/
    cache-routing machinery. ``forward`` normalizes; ``invert`` (static -- reads
    the primed buffers, no cache) denormalizes."""

    def __init__(self, d_data: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.primed = False
        self.register_buffer("mu", torch.zeros(()))
        self.register_buffer("scale", torch.ones(d_data))

    @torch.no_grad()
    def prime_normalizer(self, buffer, n: int = 20):
        x = torch.cat([next(buffer).input for _ in range(n)], dim=0)
        x = x.to(self.scale.device).detach().float()
        mu = x.mean(dim=-1, keepdim=True).mean()
        xc = x - mu
        std_e = xc.std(dim=0, unbiased=False) + self.eps  # per-feature
        std_s = ((xc / std_e).std(dim=-1, unbiased=False) + self.eps).mean()  # scalar
        self.mu.copy_(mu)
        self.scale.copy_(std_e * std_s)
        self.primed = True
        return self

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return (x - self.mu) / self.scale

    def invert(self, x, *, cache=None, **kwargs):
        return x * self.scale + self.mu


@define
class GraphNormalizer:
    """Wraps an :class:`AffineNormalizer` module; supplies the producers a
    :class:`NormalizedModel` plugs in -- the normalizer module itself and a
    stably-consed denorm op (keyed by the normalizer, not the bound method)."""

    normalizer: AffineNormalizer

    def _invert(self, x):
        # static-invertible normalizer: invert() reads primed state, not cache
        return self.normalizer.invert(x, cache=None)

    @property
    def denorm(self) -> OpRef:
        key = ("denorm", id(self.normalizer))
        return OpRef(Op.create(self._invert, "denorm", key=key))

    def wrap(self, model: SAEModel) -> "NormalizedModel":
        return NormalizedModel(
            model=model, normalize=self.normalizer, denorm=self.denorm
        )

    def denormalize(self, handle: ValueRef) -> ValueRef:
        return self.denorm(handle)

    # -- trainer-facing passthrough --
    @property
    def primed(self) -> bool:
        return self.normalizer.primed

    def prime_normalizer(self, buffer, n: int = 20):
        return self.normalizer.prime_normalizer(buffer, n=n)


class NormalizedModel(GraphSpec):
    """``normalizer[model]`` as a graph: normalize the input, run the model on it,
    denormalize the output. Exposes the normalized handles a loss reads
    (``y_norm`` / ``y_pred_norm`` / ``acts``) and the denormalized ``out``."""

    model: SAEModel = sub_slot()
    normalize: OpRef = op_slot()
    denorm: OpRef = op_slot()

    applied = model(normalize(Input))  # the model, run on the normalized input
    x_norm = normalize(Input)
    y_norm = x_norm  # autoencoding target == input
    y_pred_norm = applied.recon
    acts = applied.acts
    out = denorm(y_pred_norm)  # denormalized reconstruction (inference)
    output = out


# --- losses, as GraphSpecs -----------------------------------------------------
class GraphLoss(GraphSpec):
    """Abstract loss base. ``.on(model)`` returns the ``(model, loss_type)`` pair
    a ``@loss_prop`` declares; the trainable resolves it against the model's
    :class:`NormalizedModel`."""

    norm_model: NormalizedModel

    @classmethod
    def on(cls, model: SAEModel) -> "LossRef":
        return LossRef(model, cls)


class NormalizedReconLoss(GraphLoss):
    norm_model: NormalizedModel = sub_slot()
    loss_fn: OpRef = op_slot()
    applied = norm_model(Input)  # plug the normalized model in
    output = loss_fn(applied.y_norm, applied.y_pred_norm)


class GraphL2Loss(NormalizedReconLoss):
    loss_fn: OpRef = op_slot(default=_mse)


class GraphCosimLoss(NormalizedReconLoss):
    loss_fn: OpRef = op_slot(default=_cosim)


class GraphSparsityLoss(GraphLoss):
    norm_model: NormalizedModel = sub_slot()
    penalty: OpRef = op_slot(default=_l1)
    applied = norm_model(Input)
    output = penalty(applied.acts)


@define
class LossRef:
    """What a ``@loss_prop`` returns: which model + which loss type. The trainable
    builds the loss term against the model's :class:`NormalizedModel`."""

    model: SAEModel
    loss_type: type[GraphLoss]

    def build(self, norm_model: NormalizedModel) -> ValueRef:
        return self.loss_type(norm_model=norm_model).out


class GraphTrainable(Trainable):
    """``Trainable`` whose forward/loss are one fused interned graph.

    Init mirrors ``Trainable.__init__`` -- ``(models, resampler, losses,
    normalizer)``. ``models`` are model graphs (:class:`SAEGraphSpec`), ``losses``
    are ``dict[str, LossRef]``, ``normalizer`` is a :class:`GraphNormalizer`. It
    wraps each model in a :class:`NormalizedModel`, resolves every loss against it
    (``loss[normalizer[model]]``), adds the cosim (+ L2) metric, and runs the
    whole thing as one ``forward_values`` pass writing the cache the trainer
    reads.
    """

    def __init__(self, models, resampler, losses, normalizer: GraphNormalizer):
        cl.Module.__init__(self)
        self.normalizer = normalizer
        self.graph_models = list(models)
        main = self.graph_models[0]
        norm = {id(m): normalizer.wrap(m) for m in self.graph_models}
        main_nm = norm[id(main)]

        # resolve (model, loss_type) -> loss term over that model's NormalizedModel
        self._loss_terms = {k: lr.build(norm[id(lr.model)]) for k, lr in losses.items()}
        metric_types: dict[str, type[GraphLoss]] = {"cosim": GraphCosimLoss}
        if "L2_loss" not in losses:
            metric_types["L2_loss"] = GraphL2Loss
        self._metric_terms = {
            k: t(norm_model=main_nm).out for k, t in metric_types.items()
        }
        self._acts = main_nm.acts
        # bubble up the main model's metrics site (L0/L1) -- it lives on the
        # model's acts wherever the model put it; we collect it and apply it to
        # the rebound (normalized) acts (D-03/D-05). Fall back to computing
        # directly if the model didn't splice one.
        sites = collect([main_nm.acts], _MetricsSite)
        if sites:
            metrics = sites[0].producer.at(sites[0].inputs[0])
            self._l0, self._l1 = metrics.L0, metrics.L1
        else:
            self._l0, self._l1 = op(_l0)(main_nm.acts), op(_l1)(main_nm.acts)
        self._output = main_nm.out  # denormalized recon (inference)
        # decode: acts -> raw recon (treat acts as the input, then denormalize)
        self._decode = normalizer.denormalize(bind(main.recon, {main.acts: Input}))

        # the whole training computation as one graph; handles reached through it,
        # producers registered onto this Trainable (so .parameters()/.to() work).
        # The loss terms share this one fused pass; the (dynamic) coeff weighting
        # is applied in Python -- a graph weight source-node would capture a buffer
        # that .to(device) orphans, so weights await true graph inputs (D-04).
        self.graph = GraphSpec.of(
            output=self._output,
            acts=self._acts,
            L0=self._l0,
            L1=self._l1,
            **self._loss_terms,
            **self._metric_terms,
        )
        self.graph.register_into(self)
        # freq-tracker sites bubble up too: their stateful EMA update is deferred
        # onto each loss pass (ready for resampling; no-op if none spliced).
        self._freq_sites = self.graph.collect(FreqTracker)

        self.resampler = resampler
        resampler.assign_model(self)

    # -- training contract -----------------------------------------------------
    def loss(self, x, *, cache, y=None, coeffs=None):
        if isinstance(x, SAETrainBatch):
            x = x.input
        coeffs = dict(coeffs) if coeffs is not None else {}
        with self.graph.forward_values(x, cache=cache) as save:
            terms = {k: save(t) for k, t in self._loss_terms.items()}
            for k, t in self._metric_terms.items():
                setattr(cache, k, save(t))
            cache.L0, cache.L1 = save(self.graph.L0), save(self.graph.L1)
            for site in self._freq_sites:  # deferred EMA update after the pass
                ft = site.producer
                ft.update(ft.updated(site.inputs[0]))
        # the loss terms shared one fused pass; weight + sum in Python (coeffs are
        # dynamic). cache.<lossname> = unweighted (logging), cache.loss = total.
        total = reduce(
            lambda a, b: a + b, (m * coeffs.get(k, 1.0) for k, m in terms.items())
        )
        for k, m in terms.items():
            setattr(cache, k, m)
        cache.loss = total.item()
        return total

    def get_losses_and_metrics_names(self) -> list[str]:
        return (
            list(self._loss_terms)
            + list(self._metric_terms)
            + ["L0", "L1"]
            + ["below_3e-5", "below_1e-5", "below_3e-6", "below_1e-6", "below_1e-8"]
        )

    @make_cache_optional
    def forward(self, x, *, cache):
        if isinstance(x, SAETrainBatch):
            x = x.input
        x, shape = self.rearrange(x)
        return self.dearrange(self.graph.forward(x, cache=cache), shape)

    @make_cache_optional
    def get_acts(self, x, cache=None, pre_acts=False):
        x, shape = self.rearrange(x)
        return self.dearrange(Graph(self._acts)(x, cache=cache), shape)

    @make_cache_optional
    def encode(self, x, cache=None):
        return self.get_acts(x, cache=cache)

    @make_cache_optional
    def decode(self, x, cache=None):
        x, shape = self.rearrange(x)
        return self.dearrange(Graph(self._decode)(x, cache=cache), shape)
