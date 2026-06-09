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

from functools import cached_property
from typing import Any, Protocol

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
    deferred_execution,
    field,
    op,
    ref,
)
from saeco_research.cgraph2.dictbatch import DictBatch
from saeco_research.cgraph2.spec import (
    Arg,
    GraphSpec,
    Slot,
    SubSlots,
    arg,
    handle,
    module_slot,
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

    acts: Arg = arg()
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
    rate. ``updated(acts)`` is the handle for the new freqs given this step's acts;
    ``update(new)`` writes them back. ``update`` is a :func:`deferred_execution`
    *training callback* -- written inline as ``ft.update(ft.updated(acts))``, it
    materializes after the pass (so the handle it reads is the *pre*-update state)
    and is skipped on eval, so validation passes don't advance the EMA. The
    principled before/after split: ``freqs`` and ``updated`` are distinct nodes;
    only a training update advances the state."""

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

    @deferred_execution
    @torch.no_grad()
    def update(self, new_freqs) -> None:
        # inside a forward_values block this defers to block exit (new_freqs, a
        # handle, is materialized first) and only fires on a training pass.
        self.freqs.copy_(new_freqs)


class SAEModel(Protocol):
    """Typing protocol for a model graph: ``acts`` / ``recon`` handles, and
    appliable to an input (``model(x)``). Any :class:`SAEGraphSpec` satisfies it --
    the metaclass binds the handles per instance. Declared as *read-only* properties
    so a spec whose handle is a ``ValueRef`` subtype (e.g. a slot application) still
    satisfies the protocol (a mutable attribute would demand the exact type)."""

    @property
    def acts(self) -> ValueRef: ...
    @property
    def recon(self) -> ValueRef: ...

    def __call__(self, x: ValueRef) -> "SAEModel": ...


class SAEGraphSpec(GraphSpec):
    """Base for SAE model graphs. Subclasses define ``acts`` / ``recon`` /
    ``output`` handles; this adds the model-level hook the framework needs."""

    # The model contract, declared with `handle()` -- a typed instance-attribute
    # handle (so `g.acts` / `g.recon` type-check and an SAEGraphSpec satisfies the
    # `SAEModel` protocol) that is `init=False`, so `dataclass_transform` does NOT
    # read it as a required construction field. Subclasses just wire `recon =
    # decoder(acts)` (no annotation needed); the metaclass binds them per instance.
    acts: ValueRef = handle()
    recon: ValueRef = handle()

    name = None

    def set_to_aux_model(self, name: str):
        self.name = name
        return self


_Aggregation = GNConfig.Aggregation
_SAggregation = GNConfig.SAggregation


class AffineNormalizer(cl.Module):
    """A clean, static, invertible affine normalizer driven by a ``GNConfig``.

    Collapses saeco's ``StaticInvertibleGeneralizedNormalizer`` to the single
    affine map it actually *is*: ``forward = (x - shift) / scale``, ``invert =
    x * scale + shift``, both reading two primed buffers -- no aggregation-enum
    dispatch and no cache routing at run time. (The buffers are named ``scale`` and
    ``shift`` so the graph reads them directly via the module slot -- D-14.)

    Priming reproduces ``GeneralizedNormalizer``'s sequential estimation exactly
    (``mu_e -> mu_s -> std_e -> std_s``, each computed on the running-normalized
    data), then folds the four factors into the two buffers::

        shift = mu_e + mu_s        # per-feature offset + scalar offset
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

    scale: torch.Tensor  # primed buffers (annotated so the graph reads them typed)
    shift: torch.Tensor

    def __init__(self, d_data: int, cfg: GNConfig | None = None, eps: float = 1e-7):
        super().__init__()
        self.cfg = cfg if cfg is not None else GNConfig()
        self._require_static(self.cfg)
        self.eps = eps
        self.primed = False
        self.register_buffer("shift", torch.zeros(d_data))
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

        shift = (mu_e + mu_s).reshape(-1)
        scale = (std_e * std_s).reshape(-1)
        self.shift.copy_(shift.expand_as(self.shift) if shift.numel() == 1 else shift)
        self.scale.copy_(scale.expand_as(self.scale) if scale.numel() == 1 else scale)
        self.primed = True
        return self

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return (x - self.shift) / self.scale

    def invert(self, x, *, cache=None, **kwargs):
        return x * self.scale + self.shift


class Denormalizer(GraphSpec):
    """The inverse affine map as an *embeddable* sub-graph: ``value * scale +
    shift`` (== :meth:`AffineNormalizer.invert`). A normalizer embeds it *partially
    applied* -- ``scale``/``shift`` closed over its primed factors, ``value`` left
    open -- so a model can denormalize its output through the normalizer
    (``nx.denormalize(recon).output``) without ever naming scale/shift (D-12)."""

    value: Arg = arg()
    scale: Arg = arg()
    shift: Arg = arg()
    output = value * scale + shift


class GraphNormalizer(GraphSpec):
    """``normalize`` plus its encapsulated inverse, as a graph (D-12, D-14).

    The :class:`AffineNormalizer` is plugged in via a ``module_slot()`` and handed to
    the body as a :class:`ModuleRef` typed over the module, so access is checked against
    it. ``normalizer(Input)`` is its forward (normalized output; a ``cl.Module`` node
    so its primed buffers register + cache); ``normalizer.get(lambda n: n.scale)`` /
    ``.get(lambda n: n.shift)`` read those primed buffers live as *typed* refs
    (``ValueRef[Tensor]``; 0-ary, consed by the module's identity so forward + the
    inverse fuse). ``denormalize`` is a :class:`Denormalizer` partial over scale/shift
    with ``value`` left open. A :class:`NormalizedModel` plugs this into a ``sub_slot``
    and completes the partial with the model's recon -- so the normalizer owns its own
    inverse and the model never references scale/shift. Build it with
    ``GraphNormalizer(normalizer=norm)``; priming delegates to the bound
    :class:`AffineNormalizer`."""

    normalizer: Slot[AffineNormalizer] = module_slot()
    normalized = normalizer(Input)
    scale = normalizer.get(lambda n: n.scale)
    shift = normalizer.get(lambda n: n.shift)
    denormalize = Denormalizer.partial(scale=scale, shift=shift)
    output = normalized

    def wrap[T: SAEModel](self, model: T) -> "NormalizedModel[T]":
        """``normalizer[model]`` -- the model normalized end-to-end."""
        return NormalizedModel(model=model, normalizer=self)

    def denormalize_value(self, value: ValueRef) -> ValueRef:
        """Complete the embedded inverse on ``value`` -- the standalone form of
        ``nx.denormalize(value).output`` (used for the trainable's decode path).
        Reads the completed graph's ``.out`` (the typed output property)."""
        return self.denormalize(value).out

    # -- trainer-facing passthrough to the bound AffineNormalizer (a child) --
    @property
    def primed(self) -> bool:
        return self.normalizer.primed

    def prime_normalizer(self, buffer, n: int = 20):
        return self.normalizer.prime_normalizer(buffer, n=n)


class SAEInputs(DictBatch):
    """The model's boundary input contract: the source activations, the recon target
    (== source for an autoencoder, != for a transcoder), and the per-loss ``weights``.
    Read via the typed ``field(SAEInputs.input)`` -- no string keys (D-09). The
    *dynamic* per-loss weights live in one ``weights`` dict (keyed by the loss's own
    name), so the total-loss fold indexes ``weights[name]`` rather than synthesizing a
    stringly-typed ``w_<name>`` field per loss."""

    input: torch.Tensor
    target: torch.Tensor
    weights: dict[str, float]


class NormalizedModel[T: SAEModel](GraphSpec):
    """``normalizer[model]`` as a graph: normalize the input, run the model on it,
    denormalize the output. Exposes the normalized handles a loss reads
    (``y_norm`` / ``y_pred_norm`` / ``acts``) and the denormalized ``out``. Generic
    in the wrapped model type ``T`` so ``nm.model`` is typed and a ``@model_prop``
    can return ``NormalizedModel[T]`` (the normalized wrapper) rather than ``T``."""

    model: T = sub_slot()
    normalizer: GraphNormalizer = sub_slot()

    x_in = field(SAEInputs.input)  # typed boundary fields: source + recon target
    y_in = field(SAEInputs.target)
    nx = normalizer(x_in)  # the normalizer, applied to the input
    applied = model(nx.normalized)  # the model, run on the normalized input
    x_norm = nx.normalized
    y_norm = normalizer(y_in).normalized  # transcoder: target may differ from input
    y_pred_norm = applied.recon
    acts = applied.acts
    out = nx.denormalize(applied.recon).output  # the normalizer's own inverse
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
    loss_fn: Slot = op_slot()
    applied = norm_model(Input)  # plug the normalized model in
    output = loss_fn(applied.y_norm, applied.y_pred_norm)


class GraphL2Loss(NormalizedReconLoss):
    loss_fn: Slot = op_slot(default=_mse)


class GraphCosimLoss(NormalizedReconLoss):
    loss_fn: Slot = op_slot(default=_cosim)


class GraphSparsityLoss(GraphLoss):
    norm_model: NormalizedModel = sub_slot()
    penalty: Slot = op_slot(default=_l1)
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


def _weight(weights: ValueRef, name: str) -> ValueRef[float]:
    """The scalar weight for loss ``name``, read from the seeded ``weights`` dict
    (``weights[name]``). Consed by ``name`` -- two reads of the same weight fuse, and
    distinct weights stay distinct -- so the weight stays a *graph input* (seeded each
    step, device-clean: D-04), not a captured constant, with no ``w_<name>`` field."""
    reader = Op.create(
        lambda w: w[name], label=f"weights[{name}]", key=("weight", name)
    )
    return ref(reader, (weights,))


class TrainGraph(GraphSpec):
    """The trainable's outer computation, as a (mostly static) class block.

    ``total_loss`` (the backprop target) is a *declarative* weighted fold over a
    **dynamic** set of loss sub-graphs: ``Σ weights[name] · loss.out``, reduced by
    ``reduce_sum`` into one flat N-ary node (``weights`` is one seeded dict field,
    indexed by each loss's own name). ``output`` is the main model's denormalized
    recon (inference). The per-loss / metric / L0 / L1 *logging* values are looked up
    dynamically in ``GraphTrainable.loss`` (not declared here) -- so this class stays
    static while being parameterized by the dynamic losses."""

    losses: SubSlots[GraphLoss] = sub_slots()
    main: NormalizedModel = sub_slot()

    total_loss = losses.reduce(
        # field(SAEInputs.weights) interns to one shared node; each loss reads its
        # own weights[name] off it.
        lambda name, loss: _weight(field(SAEInputs.weights), name) * loss.out,
        reduce_sum,
    )
    output = main(Input).out  # the main model reads its own fields; Input is a no-op

    # -- the logging handles, derived from the bound losses/main (cached). The
    # trainable `save()`s these in its forward_values pass; deriving them here keeps
    # the assembly with the structure that has the losses + main, not in the
    # trainable's __init__.
    @cached_property
    def loss_terms(self) -> dict[str, ValueRef]:
        """Per-loss (unweighted) value handles -- what the trainer logs."""
        return {k: loss.out for k, loss in self.losses.items()}

    @cached_property
    def metric_terms(self) -> dict[str, ValueRef]:
        """Metric handles over the main model: cosim, plus L2 if it isn't a loss."""
        types: dict[str, type[GraphLoss]] = {"cosim": GraphCosimLoss}
        if "L2_loss" not in self.losses:
            types["L2_loss"] = GraphL2Loss
        return {k: t(norm_model=self.main).out for k, t in types.items()}

    @cached_property
    def _l0_l1(self) -> tuple[ValueRef, ValueRef]:
        # bubble up the main model's metrics site (D-03/D-05), applied to the
        # rebound (normalized) acts; fall back to computing directly if no site.
        sites = collect([self.main.acts], _MetricsSite)
        if sites:
            m = sites[0].producer.at(sites[0].inputs[0])
            return m.L0, m.L1
        return op(_l0)(self.main.acts), op(_l1)(self.main.acts)

    @property
    def l0(self) -> ValueRef:
        return self._l0_l1[0]

    @property
    def l1(self) -> ValueRef:
        return self._l0_l1[1]


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
        # NormalizedModel (kept whole so TrainGraph can fold them + derive the
        # per-loss logging handles); the trainable just builds + runs the graph.
        self._loss_graphs: dict[str, GraphLoss] = {
            k: lr.build_graph(norm[id(lr.model)]) for k, lr in losses.items()
        }
        self._acts: ValueRef = main_nm.acts  # inference: encode
        self._output: ValueRef = main_nm.out  # inference: denormalized recon
        # decode: acts -> raw recon (treat acts as the input, then denormalize)
        self._decode: ValueRef = normalizer.denormalize_value(
            bind(main.recon, {main.acts: Input})
        )

        # the outer graph as a (mostly static) class block: `total_loss` is the
        # declarative weighted fold over the dynamic loss sub-graphs, `output` is
        # the main recon, and the logging handles (loss_terms/metric_terms/l0/l1)
        # are its cached properties. register_into over it covers every producer
        # (total_loss's cone is all the loss sub-graphs); GraphSpec is a plain
        # object (not an nn.Module), so register_into does the bookkeeping.
        self.graph: TrainGraph = TrainGraph(
            losses=SubSlots.from_dict(self._loss_graphs), main=main_nm
        )
        self.graph.register_into(self)
        # freq-tracker sites bubble up too: their stateful EMA update is deferred
        # onto each loss pass (ready for resampling; no-op if none spliced).
        # typed via the generic collect: site.producer is a FreqTracker
        self._freq_sites: list[ValueRef[Any, FreqTracker]] = self.graph.collect(
            FreqTracker
        )

        self.resampler: Resampler = resampler
        resampler.assign_model(self)

    # -- training contract -----------------------------------------------------
    def _batch(self, x, y=None, coeffs=None) -> SAEInputs:
        """The boundary input batch: the typed source + recon target, plus the per-loss
        ``weights`` dict (keyed by loss name) seeded from ``coeffs``. The target
        defaults to the source (autoencoder), but an ``SAETrainBatch`` may carry
        distinct ``target_sites`` (transcoder/crosscoder) -- honor those when no
        explicit ``y`` is given; an explicit ``y`` always wins."""
        target = x.target if isinstance(x, SAETrainBatch) else x
        if isinstance(x, SAETrainBatch):
            x = x.input
        if isinstance(y, SAETrainBatch):
            y = y.input
        if y is not None:
            target = y
        coeffs = coeffs or {}
        weights = {k: coeffs.get(k, 1.0) for k in self._loss_graphs}
        return SAEInputs(input=x, target=target, weights=weights)

    def loss(self, x, *, cache, y=None, coeffs=None):
        # training=self.training gates the freq-tracker EMA: an eval-mode loss pass
        # (Trainer.eval_step under .eval()) won't advance the dead-feature freqs.
        with self.graph.forward_values(
            self._batch(x, y, coeffs), cache=cache, training=self.training
        ) as save:
            total = save(self.graph.total_loss)  # the one static backprop handle
            # dynamic lookups: the per-loss / metric / L0 / L1 handles aren't on the
            # (mostly-static) graph -- they're saved here by handle, sharing the one
            # fused memoized pass (the model is computed once and reused).
            logged = {k: save(t) for k, t in self.graph.loss_terms.items()}  # unwtd
            logged.update({k: save(t) for k, t in self.graph.metric_terms.items()})
            l0, l1 = save(self.graph.l0), save(self.graph.l1)
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
            list(self.graph.loss_terms)
            + list(self.graph.metric_terms)
            + ["L0", "L1"]
            + [name for name, _ in _BELOW]
        )

    @make_cache_optional
    def forward(self, x, *, cache):
        if isinstance(x, SAETrainBatch):
            x = x.input
        x, shape = self.rearrange(x)
        # inference reads SAEInputs.input; target/weights are unused on the out path
        batch = SAEInputs(input=x, target=x, weights={})
        return self.dearrange(self.graph.forward(batch, cache=cache), shape)

    @make_cache_optional
    def get_acts(self, x, cache=None, pre_acts=False):
        x, shape = self.rearrange(x)
        batch = SAEInputs(input=x, target=x, weights={})
        return self.dearrange(Graph(self._acts)(batch, cache=cache), shape)

    @make_cache_optional
    def encode(self, x, cache=None):
        return self.get_acts(x, cache=cache)

    @make_cache_optional
    def decode(self, x, cache=None):
        x, shape = self.rearrange(x)
        return self.dearrange(Graph(self._decode)(x, cache=cache), shape)
