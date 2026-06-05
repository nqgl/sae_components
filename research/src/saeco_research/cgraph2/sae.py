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

from typing import Protocol

import torch
from attrs import define

import saeco.core as cl
from saeco.components.resampling import Resampler
from saeco.data.training_data.sae_train_batch import SAETrainBatch
from saeco.trainer.normalizers.normalizer import GNConfig
from saeco.trainer.trainable import Trainable, make_cache_optional
from saeco_research.cgraph2.components import reduce_sum
from saeco_research.cgraph2.core import (
    Graph,
    Input,
    Op,
    OpRef,
    ValueRef,
    bind,
    collect,
    field,
    op,
    ref,
)
from saeco_research.cgraph2.dictbatch import DictBatch
from saeco_research.cgraph2.spec import (
    GraphSpec,
    SubSlots,
    arg,
    op_slot,
    sub_slot,
    sub_slots,
)


# --- graph-leaf math (module-level so consing keys on function structure) -----
def _mse(a, b):
    return ((a - b) ** 2).mean()


def _l0(a):
    return (a != 0).sum(-1).float().mean(0).sum()


def _l1(a):
    return a.abs().mean(0).sum()


def _cosim(a, b):
    return torch.cosine_similarity(a, b, dim=-1).mean()


# the freq thresholds the trainer logs (fraction of features below each freq)
_BELOW = (
    ("below_3e-5", 3e-5),
    ("below_1e-5", 1e-5),
    ("below_3e-6", 3e-6),
    ("below_1e-6", 1e-6),
    ("below_1e-8", 1e-8),
)


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


_Aggregation = GNConfig.Aggregation
_SAggregation = GNConfig.SAggregation


class AffineNormalizer(cl.Module):
    """A clean, static, invertible affine normalizer driven by a ``GNConfig``.

    Collapses saeco's ``StaticInvertibleGeneralizedNormalizer`` to the single
    affine map it actually *is*: ``forward = (x - mu) / scale``, ``invert =
    x * scale + mu``, both reading two primed buffers -- no aggregation-enum
    dispatch and no cache routing at run time.

    Priming reproduces ``GeneralizedNormalizer``'s sequential estimation exactly
    (``mu_e -> mu_s -> std_e -> std_s``, each computed on the running-normalized
    data), then folds the four factors into the two buffers::

        mu    = mu_e + mu_s        # per-feature offset + scalar offset
        scale = std_e * std_s      # per-feature scale  * scalar scale

    Only the *static* aggregations ``{DONTUSE, PRIMED}`` are supported (each term
    is either off or primed-from-data) -- that's exactly the set
    ``StaticInvertibleGeneralizedNormalizer`` allows for ``mu_s``/``std_s`` and a
    static subset for ``mu_e``/``std_e``. The dynamic ones (RUNNING_AVG /
    BATCH_AVG / LEARNED / SAMPLE) and ``sandwich`` aren't a single static affine
    map and raise ``NotImplementedError`` pointing here; use the stock
    ``GeneralizedNormalizer`` for those.

    Verified value-identical to the stock normalizer across all 16 static
    ``{DONTUSE, PRIMED}`` configs. Operates on batched ``(..., d)`` activations
    (what the training pipeline always feeds, post-``rearrange``) and returns the
    same shape -- the buffers are folded to ``(d,)``, so unlike the stock
    normalizer it does *not* introduce a spurious leading singleton dim for a bare
    unbatched ``(d,)`` input (the values are identical either way).
    """

    def __init__(self, d_data: int, cfg: GNConfig | None = None, eps: float = 1e-7):
        super().__init__()
        self.cfg = cfg if cfg is not None else GNConfig()
        self._require_static(self.cfg)
        self.eps = eps
        self.primed = False
        self.register_buffer("mu", torch.zeros(d_data))
        self.register_buffer("scale", torch.ones(d_data))

    @staticmethod
    def _require_static(cfg: GNConfig) -> None:
        static_e = (_Aggregation.DONTUSE, _Aggregation.PRIMED)
        static_s = (_SAggregation.DONTUSE, _SAggregation.PRIMED)
        for name, val, ok in (
            ("mu_e", cfg.mu_e, static_e),
            ("std_e", cfg.std_e, static_e),
            ("mu_s", cfg.mu_s, static_s),
            ("std_s", cfg.std_s, static_s),
        ):
            if val not in ok:
                raise NotImplementedError(
                    f"AffineNormalizer supports only the static aggregations "
                    f"{{DONTUSE, PRIMED}}; {name}={val!r} is dynamic/stateful and "
                    f"isn't a single static affine map (see AffineNormalizer in "
                    f"cgraph2/sae.py) -- use the stock GeneralizedNormalizer."
                )
        if cfg.sandwich:
            raise NotImplementedError(
                "AffineNormalizer does not support sandwich=True (not a single "
                "static affine map); see AffineNormalizer in cgraph2/sae.py."
            )

    @torch.no_grad()
    def prime_normalizer(self, buffer, n: int = 20):
        cfg = self.cfg
        x = torch.cat([next(buffer).input for _ in range(n)], dim=0)
        x = x.to(self.scale.device).detach().float()
        zero = torch.zeros((), device=x.device)
        one = torch.ones((), device=x.device)

        # Sequential, in GeneralizedNormalizer's order: each factor is estimated
        # on the data already normalized by the previous factors.
        # mu_e: per-feature mean over the batch dim.
        mu_e = x.mean(dim=0, keepdim=True) if cfg.mu_e == _Aggregation.PRIMED else zero
        x = x - mu_e
        # mu_s: scalar -- per-sample (token) mean, averaged.
        if cfg.mu_s == _SAggregation.PRIMED:
            mu_s = x.mean(dim=-1, keepdim=True).mean()
        else:
            mu_s = zero
        x = x - mu_s
        # std_e: per-feature std over the batch dim. Matches GeneralizedNormalizer's
        # std_e(): abs(elementwise_std + eps) + 1e-7.
        if cfg.std_e == _Aggregation.PRIMED:
            std_e = (x.std(dim=0, keepdim=True, unbiased=False) + self.eps).abs() + 1e-7
        else:
            std_e = one
        x = x / std_e
        # std_s: scalar std.
        if cfg.std_s == _SAggregation.PRIMED:
            std_s = (x.std(dim=-1, keepdim=True, unbiased=False) + self.eps).mean()
        else:
            std_s = one

        mu = (mu_e + mu_s).reshape(-1)
        scale = (std_e * std_s).reshape(-1)
        self.mu.copy_(mu.expand_as(self.mu) if mu.numel() == 1 else mu)
        self.scale.copy_(scale.expand_as(self.scale) if scale.numel() == 1 else scale)
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


class SAEInputs(DictBatch):
    """The model's boundary input contract: the source activations and the recon
    target (== source for an autoencoder, != for a transcoder). Read via the typed
    ``field(SAEInputs.input)`` -- no string keys (D-09). Per-loss weight fields
    (``w_<lossname>``) are the *dynamic* tail (they vary with the loss set), kept as
    derived names rather than declared here."""

    input: torch.Tensor
    target: torch.Tensor


class NormalizedModel(GraphSpec):
    """``normalizer[model]`` as a graph: normalize the input, run the model on it,
    denormalize the output. Exposes the normalized handles a loss reads
    (``y_norm`` / ``y_pred_norm`` / ``acts``) and the denormalized ``out``."""

    model: SAEModel = sub_slot()
    normalize: OpRef = op_slot()
    denorm: OpRef = op_slot()

    x_in = field(SAEInputs.input)  # typed boundary fields: source + recon target
    y_in = field(SAEInputs.target)
    applied = model(normalize(x_in))  # the model, run on the normalized input
    x_norm = normalize(x_in)
    y_norm = normalize(y_in)  # transcoder: target may differ from input (else == x)
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

    def build_graph(self, norm_model: NormalizedModel) -> GraphLoss:
        return self.loss_type(norm_model=norm_model)

    def build(self, norm_model: NormalizedModel) -> ValueRef:
        return self.build_graph(norm_model).out


class TrainGraph(GraphSpec):
    """The trainable's outer computation, as a (mostly static) class block.

    ``total_loss`` (the backprop target) is a *declarative* weighted fold over a
    **dynamic** set of loss sub-graphs: ``Σ field("w_<k>") · loss.out``, reduced by
    ``reduce_sum`` into one flat N-ary node. ``output`` is the main model's
    denormalized recon (inference). The per-loss / metric / L0 / L1 *logging* values
    are looked up dynamically in ``GraphTrainable.loss`` (not declared here) -- so
    this class stays static while being parameterized by the dynamic losses."""

    losses: SubSlots[GraphLoss] = sub_slots()
    main: NormalizedModel = sub_slot()

    total_loss = losses.reduce(
        lambda name, loss: field(f"w_{name}") * loss.out, reduce_sum
    )
    output = main(Input).out  # the main model reads its own fields; Input is a no-op


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

    def __init__(
        self,
        models: list[SAEGraphSpec],
        resampler: Resampler,
        losses: dict[str, LossRef],
        normalizer: GraphNormalizer,
    ) -> None:
        cl.Module.__init__(self)
        if not losses:
            raise ValueError("GraphTrainable needs at least one loss")
        self.normalizer: GraphNormalizer = normalizer
        self.graph_models: list[SAEGraphSpec] = list(models)
        main: SAEGraphSpec = self.graph_models[0]
        norm: dict[int, NormalizedModel] = {
            id(m): normalizer.wrap(m) for m in self.graph_models
        }
        main_nm: NormalizedModel = norm[id(main)]

        # resolve (model, loss_type) -> a loss *sub-graph* over that model's
        # NormalizedModel (kept whole so TrainGraph can fold them); `.out` is the
        # per-loss value handle we log.
        self._loss_graphs: dict[str, GraphLoss] = {
            k: lr.build_graph(norm[id(lr.model)]) for k, lr in losses.items()
        }
        self._loss_terms: dict[str, ValueRef] = {
            k: g.out for k, g in self._loss_graphs.items()
        }
        metric_types: dict[str, type[GraphLoss]] = {"cosim": GraphCosimLoss}
        if "L2_loss" not in losses:
            metric_types["L2_loss"] = GraphL2Loss
        self._metric_terms: dict[str, ValueRef] = {
            k: t(norm_model=main_nm).out for k, t in metric_types.items()
        }
        self._acts: ValueRef = main_nm.acts
        # bubble up the main model's metrics site (L0/L1) -- it lives on the
        # model's acts wherever the model put it; we collect it and apply it to
        # the rebound (normalized) acts (D-03/D-05). Fall back to computing
        # directly if the model didn't splice one.
        self._l0: ValueRef
        self._l1: ValueRef
        sites = collect([main_nm.acts], _MetricsSite)
        if sites:
            metrics = sites[0].producer.at(sites[0].inputs[0])
            self._l0, self._l1 = metrics.L0, metrics.L1
        else:
            self._l0, self._l1 = op(_l0)(main_nm.acts), op(_l1)(main_nm.acts)
        self._output: ValueRef = main_nm.out  # denormalized recon (inference)
        # decode: acts -> raw recon (treat acts as the input, then denormalize)
        self._decode: ValueRef = normalizer.denormalize(
            bind(main.recon, {main.acts: Input})
        )

        # the outer graph as a (mostly static) class block: `total_loss` is the
        # declarative weighted fold over the dynamic loss sub-graphs, `output` is
        # the main recon. register_into over it covers every producer (total_loss's
        # cone is all the loss sub-graphs). GraphSpec is a plain object (not an
        # nn.Module), so register_into does the param/buffer bookkeeping explicitly.
        self.graph: TrainGraph = TrainGraph(
            losses=SubSlots.from_dict(self._loss_graphs), main=main_nm
        )
        self._total_loss: ValueRef = self.graph.total_loss
        self.graph.register_into(self)
        # freq-tracker sites bubble up too: their stateful EMA update is deferred
        # onto each loss pass (ready for resampling; no-op if none spliced).
        self._freq_sites: list[ValueRef] = self.graph.collect(FreqTracker)

        self.resampler: Resampler = resampler
        resampler.assign_model(self)

    # -- training contract -----------------------------------------------------
    def _batch(self, x, y=None, coeffs=None) -> SAEInputs:
        """The boundary input batch: the typed source + recon target (==source for
        an autoencoder), plus a per-loss weight field (the dynamic ``w_<k>`` tail)
        seeded from ``coeffs``."""
        if isinstance(x, SAETrainBatch):
            x = x.input
        if isinstance(y, SAETrainBatch):
            y = y.input
        coeffs = coeffs or {}
        batch = SAEInputs(input=x, target=x if y is None else y)
        for k in self._loss_terms:  # dynamic per-loss weight fields
            setattr(batch, f"w_{k}", coeffs.get(k, 1.0))
        return batch

    def loss(self, x, *, cache, y=None, coeffs=None):
        with self.graph.forward_values(self._batch(x, y, coeffs), cache=cache) as save:
            total = save(self.graph.total_loss)  # the one static backprop handle
            # dynamic lookups: the per-loss / metric / L0 / L1 handles aren't on the
            # (mostly-static) graph -- they're saved here by handle, sharing the one
            # fused memoized pass (the model is computed once and reused).
            logged = {k: save(t) for k, t in self._loss_terms.items()}  # unweighted
            logged.update({k: save(t) for k, t in self._metric_terms.items()})
            l0, l1 = save(self._l0), save(self._l1)
            for site in self._freq_sites:  # deferred EMA update after the pass
                ft = site.producer
                ft.update(ft.updated(site.inputs[0]))
        # cache.<lossname>/<metric> = unweighted (logging); cache.loss = the total.
        for k, m in logged.items():
            setattr(cache, k, m)
        cache.L0, cache.L1 = l0, l1
        cache.loss = total.item()
        # below_* freq metrics, from the (now-updated) bubbled-up freq trackers
        for site in self._freq_sites:
            freqs = site.producer.freqs
            for name, thr in _BELOW:
                setattr(cache, name, (freqs < thr).float().mean())
        return total

    def get_losses_and_metrics_names(self) -> list[str]:
        return (
            list(self._loss_terms)
            + list(self._metric_terms)
            + ["L0", "L1"]
            + [name for name, _ in _BELOW]
        )

    @make_cache_optional
    def forward(self, x, *, cache):
        if isinstance(x, SAETrainBatch):
            x = x.input
        x, shape = self.rearrange(x)
        # inference reads SAEInputs.input; target is unused on the out path (== x)
        batch = SAEInputs(input=x, target=x)
        return self.dearrange(self.graph.forward(batch, cache=cache), shape)

    @make_cache_optional
    def get_acts(self, x, cache=None, pre_acts=False):
        x, shape = self.rearrange(x)
        batch = SAEInputs(input=x, target=x)
        return self.dearrange(Graph(self._acts)(batch, cache=cache), shape)

    @make_cache_optional
    def encode(self, x, cache=None):
        return self.get_acts(x, cache=cache)

    @make_cache_optional
    def decode(self, x, cache=None):
        x, shape = self.rearrange(x)
        return self.dearrange(Graph(self._decode)(x, cache=cache), shape)
