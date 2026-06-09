"""cgraph v2 — the GraphSpec authoring surface (the only way to define graphs).

A `GraphSpec` class wires interned nodes using `Input` (the global input) and
slots. Constructing binds the slots (structural re-interning) and exposes every
named body value as a handle (a typed instance attribute).

    class SAESpec(GraphSpec):
        x: Arg = arg(default=Input)   # value slot -> the global Input
        encoder: Op = op_slot()       # producer slot
        decoder: Op = op_slot()
        acts = encoder(x)             # handle
        recon = decoder(acts)         # handle
        output = recon

Two composition features beyond plain slots:

* **Inheritance.** A subclass inherits its bases' slots/handles/output and may
  override a slot's default -- ``class L2Loss(ReconLoss): loss_fn =
  op_slot(default=_mse)``. A base with no ``output`` is *abstract* (can't be
  constructed) -- useful for a shared base that just adds methods.
* **Sub-model slots.** ``m: T = sub_slot()`` takes a whole sub-graph (anything
  with handle attributes, e.g. a ``GraphSAE``); ``m(x)`` in the body plugs it in
  with *its* input rebound to ``x``, and ``m(x).recon`` references the sub-graph's
  ``recon`` handle. The graph-native "plug a component in here" move.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Self, cast, dataclass_transform, overload

import torch
from attrs import define, field

from saeco_research.cgraph2.core import (
    Graph,
    Input,
    OpRef,
    Subs,
    ValueRef,
    _producer_label,
    _register_producers,
    _SaveSession,
    _toposort,
    bind,
    collect,
    op,
    ref,
    ref_as,
)
from saeco_research.cgraph2.core import (
    Op as _CoreOp,
)
from saeco_research.cgraph2.core import (
    _names as _name_map,
)

# The annotation alias `Arg` points at the value-slot descriptor `ArgSlot` (defined
# just below) -- so a field `x: Arg = arg()` gets the descriptor's construction face.
# Op/module/sub slots share one descriptor `Slot[T]` (defined further down).

_NODEFAULT = object()
_HANDLE_FIELD = object()  # sentinel returned by `handle()`; the metaclass ignores it


@define(eq=False)
class _SlotMarker:
    """Marker producer for an op/module slot: a per-slot identity token bound to a
    concrete producer (a ``cl.Module`` / ``Op`` / callable) at construction. The
    forward apply ``slot(x)`` is the node ``ref(marker, (x,))``; binding substitutes
    ``marker -> producer`` (generic producer-substitution in ``bind``), so no special
    apply-producer is needed. ``default`` (if set) is the producer used when the slot
    is left unbound. ``.get(fn)`` reads ``fn(producer)`` via a separate ``_SlotGet``."""

    default: object = _NODEFAULT


@define(eq=False)
class _ArgSlot:
    """Marker producer for a value slot. ``default`` is what the slot resolves to
    when left unbound (``_NODEFAULT`` == required). Each ``arg()`` is a *unique*
    marker, so binding one slot never disturbs another (or the global ``Input``)."""

    default: object = _NODEFAULT


class ArgSlot(ValueRef):
    """What ``arg()`` returns, and a descriptor (D-14). Body-time it *is* a ``ValueRef``
    (the marker node you pass into ops); its construction face (``__set__``) takes any
    source value -- a handle (``ValueRef``), a tensor, or a parameter -- which the
    metaclass wraps into a source node; instance-time (``__get__``) it's the bound
    value (arg markers are collected as handles, so the metaclass setattrs them)."""

    @overload
    def __get__(self, obj: None, objtype: Any = None) -> ArgSlot: ...
    @overload
    def __get__(self, obj: object, objtype: Any = None) -> ValueRef: ...
    def __get__(self, obj: object | None, objtype: Any = None) -> ValueRef:
        if obj is None:
            return self
        return obj.__dict__.get(f"__arg_{id(self)}", self)

    def __set__(self, obj: object, value: ValueRef | torch.Tensor) -> None:
        obj.__dict__[f"__arg_{id(self)}"] = value


# The `Arg` annotation alias points at the value-slot descriptor (see note above
# `_NODEFAULT`); `Slot[T]` (op/module/sub) is defined further down.
Arg = ArgSlot


@define(eq=False)
class _SlotHandle:
    """Producer of a ``slot(x).attr`` node -- a *named handle* of a sub-graph slot. At
    bind time (the slot bound to a sub-graph) it splices in the bound sub-graph's
    ``attr`` handle with that handle's ``Input`` rebound to ``x``. The sub-graph face
    of a slot: a leaf module has one forward value (``slot(x)``), a sub-graph has
    several named outputs (``slot(x).recon`` / ``.acts``) reached this way."""

    slot: _SlotMarker
    attr: str

    def __graph_rebind__(self, inputs: tuple[ValueRef, ...], subs: Subs) -> ValueRef:
        bound = subs.get(self.slot, None)
        if bound is None:  # slot not in subs: keep the node (re-interned inputs)
            return ref(self, inputs)
        if isinstance(bound, _SlotMarker):  # inheritance re-point: swap the marker
            return ref(_SlotHandle(bound, self.attr), inputs)
        (x_node,) = inputs  # construction: splice the bound sub-graph
        return bind(getattr(bound, self.attr), {Input: x_node})

    def __partial_call__(self, node: ValueRef, value: ValueRef):
        (x,) = node.inputs
        return _AppliedPartial(self.slot, self.attr, x, value)


class _AppliedSlot(ValueRef[torch.Tensor]):
    """The result of ``slot(x)`` -- a ``ValueRef[Tensor]`` (so a leaf module/op slot's
    forward value is a usable tensor handle) that *also* exposes ``.attr`` for a
    sub-graph slot's named handles (``slot(x).recon``). The two faces of applying a
    slot, in one node:

    * **leaf module / op** -- ``slot(x)`` is the forward value; its producer (the slot
      marker) binds to the bound module/op, so it evaluates like any node.
    * **sub-graph** -- ``slot(x).recon`` reads a handle (via ``__getattr__`` ->
      ``_SlotHandle``); the forward node itself is never used as a value, so it never
      reaches eval (a whole sub-graph isn't a single value).

    Interned (via ``ref_as``) so repeated ``slot(x)`` fuse, exactly like any node."""

    def __getattr__(self, attr: str) -> ValueRef[torch.Tensor]:
        if attr.startswith("_"):
            raise AttributeError(attr)
        return ref(_SlotHandle(self.producer, attr), self.inputs)


def _slot_get_ref(producer, fn: Callable) -> ValueRef:
    """A 0-ary node that reads ``fn(producer)`` *live* at eval (so it follows
    ``.to(device)`` / priming), consed by ``(id(producer), id(fn))``. ``fn`` is created
    once in the class body (it lives in the template), so its id is stable -- repeated
    reads and two specs over the same producer fuse; a different one doesn't."""
    reader = _CoreOp.create(lambda: fn(producer), "get", key=(id(producer), id(fn)))
    return OpRef(reader)()


@define(eq=False)
class _SlotGet:
    """Producer of ``slot.get(fn)``: at bind time, a 0-ary node reading
    ``fn(bound_producer)`` (e.g. ``lambda n: n.scale`` -- a buffer; or
    ``lambda m: m.weight`` -- an op's param). The attr-read face shared by op and
    module slots, carrying ``fn``'s return type into ``ValueRef[R]``."""

    slot: _SlotMarker
    fn: Callable

    def __graph_rebind__(self, inputs: tuple[ValueRef, ...], subs: Subs) -> ValueRef:
        bound = subs.get(self.slot, None)
        if bound is None:
            return ref(self, inputs)
        if isinstance(bound, _SlotMarker):  # inheritance re-point
            return ref(_SlotGet(bound, self.fn), inputs)
        return _slot_get_ref(bound, self.fn)


class Slot[M]:
    """What ``op_slot()`` / ``module_slot()`` / ``sub_slot()`` return: one typed
    reference to a callable+attributed component (type ``M``) plugged into a graph.
    It is **a descriptor** -- which is what reconciles the three faces a slot has:

    * *class-body time* it is itself -- ``slot(x)`` applies the component (a module's
      forward, an op's call) and ``slot.get(lambda c: c.weight)`` reads a param/buffer
      as a *typed* graph ref (the lambda is checked against the real ``M``; its return
      ``R`` carries through to ``ValueRef[R]`` -- no string keys, no type-laundering);
    * *constructed instance* ``self.slot`` is the bound component ``M`` (so a
      passthrough like ``self.norm.primed`` type-checks) -- via ``__get__``;
    * *construction* ``Spec(enc=module)`` takes the component ``M`` -- via ``__set__``
      (the metaclass's ``setattr`` of the bound component routes through it).

    One ``Slot[M]`` subsumes the former ``OpSlot`` (a producer) and ``ModuleSlot[M]``
    (a leaf module): a module and an op are the same idea -- a callable you also read
    attributes off -- so they share this descriptor. The forward apply ``slot(x)`` is
    just ``ref(marker, (x,))``; binding substitutes the marker for the bound producer
    (generic producer-substitution in ``bind``)."""

    def __init__(self, marker: _SlotMarker):
        self._marker = marker
        self._name = "slot"

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    # -- class-body proxy: build template nodes --
    def __call__(self, *x: ValueRef) -> _AppliedSlot:
        # an _AppliedSlot: a forward value for a leaf module/op (`slot(x)`), and the
        # carrier of `.attr` handle access for a sub-graph (`slot(x).recon`).
        return cast("_AppliedSlot", ref_as(_AppliedSlot, self._marker, x))

    def get[R](self, getter: Callable[[M], R]) -> ValueRef[R]:
        return cast("ValueRef[R]", ref(_SlotGet(self._marker, getter), ()))

    # -- descriptor: class access -> the slot; instance access -> the bound component
    @overload
    def __get__(self, obj: None, objtype: Any = None) -> Slot[M]: ...
    @overload
    def __get__(self, obj: object, objtype: Any = None) -> M: ...
    def __get__(self, obj: object | None, objtype: Any = None) -> Slot[M] | M:
        if obj is None:
            return self
        return obj.__dict__[f"__slot_{self._name}"]

    def __set__(self, obj: object, value: M) -> None:
        obj.__dict__[f"__slot_{self._name}"] = value


@define(eq=False)
class _PartialComplete:
    slot: _SlotMarker
    attr: str
    handle: str

    def __graph_rebind__(self, inputs: tuple[ValueRef, ...], subs: Subs) -> ValueRef:
        x_node, value_node = inputs
        bound = subs.get(self.slot, None)
        if bound is None:  # outer slot not bound here: keep the deferred node
            return ref(self, inputs)
        if isinstance(bound, _SlotMarker):  # inheritance re-point: swap the marker
            return ref(_PartialComplete(bound, self.attr, self.handle), inputs)
        partial = getattr(bound, self.attr)
        rebound = partial.rebind({Input: x_node})
        completed = rebound(value_node)
        return getattr(completed, self.handle)


@define(eq=False)
class _AppliedPartial:
    slot: _SlotMarker
    attr: str
    x: ValueRef
    value: ValueRef

    def __getattr__(self, handle_name: str) -> ValueRef:
        if handle_name.startswith("_"):
            raise AttributeError(handle_name)
        return ref(
            _PartialComplete(self.slot, self.attr, handle_name), (self.x, self.value)
        )


@define(eq=False)
class _SubSlots:
    """Marker for a *variadic* sub-model slot, bound to a ``dict`` of sub-graphs."""


@define(eq=False)
class _MapReduceOverSlots:
    """Producer of a deferred fold over a variadic sub-slot. At construction (when
    the slot is bound to a concrete ``dict`` of sub-graphs) it runs
    ``per_item(key, subgraph)`` for each and combines the terms with ``combine`` --
    the declarative way to map/reduce a *dynamic* collection in a class body (the
    body can't iterate a slot that's only bound later, so the iteration is deferred
    here and re-interned at construction, staying flat/fused)."""

    slots: _SubSlots
    per_item: Callable
    combine: Callable

    def __graph_rebind__(self, inputs: tuple[ValueRef, ...], subs: Subs) -> ValueRef:
        bound = subs.get(self.slots, None)
        if bound is None:  # slot not bound here: keep the deferred node
            return ref(self, inputs)
        return self.combine([self.per_item(k, g) for k, g in bound.items()])


@define(eq=False)
class SubSlots[T]:
    """The typed annotation for a ``sub_slots()`` field (``losses: SubSlots[
    GraphLoss]``). Generic in the sub-graph type ``T`` so ``losses.reduce`` is bound
    and ``per_item``'s ``loss`` is typed ``T``.

    In a class body it's the variadic-slot *marker*; ``reduce(per_item, combine)``
    builds a deferred fold over the (eventually-bound) dict of sub-graphs --
    ``per_item(key, subgraph)`` makes each term, ``combine`` (e.g. ``reduce_sum``)
    folds them into one flat node. ``SubSlots.from_dict(d)`` wraps a concrete dict
    so it can be *bound* to the slot at construction without a cast (the metaclass
    also accepts a bare dict)."""

    slots: _SubSlots = field(factory=_SubSlots)
    bound: dict[str, T] | None = field(default=None)

    @classmethod
    def from_dict(cls, subgraphs: dict[str, T]) -> SubSlots[T]:
        """Wrap a concrete dict of sub-graphs for binding to a ``sub_slots()`` field
        (a typed alternative to passing the bare dict)."""
        return cls(bound=dict(subgraphs))

    def reduce(
        self,
        per_item: Callable[[str, T], ValueRef],
        combine: Callable[[list[ValueRef]], ValueRef],
    ) -> ValueRef:
        return ref(_MapReduceOverSlots(self.slots, per_item, combine), ())

    # -- descriptor (D-14): body-time the marker (``losses.reduce``), instance-time the
    # bound ``dict`` (``g.losses["L2_loss"]``), construction takes ``SubSlots | dict``.
    @overload
    def __get__(self, obj: None, objtype: Any = None) -> SubSlots[T]: ...
    @overload
    def __get__(self, obj: object, objtype: Any = None) -> dict[str, T]: ...
    def __get__(self, obj: object | None, objtype: Any = None) -> SubSlots[T] | dict:
        if obj is None:
            return self
        return obj.__dict__[f"__subslots_{id(self)}"]

    def __set__(self, obj: object, value: SubSlots[T] | dict[str, T]) -> None:
        obj.__dict__[f"__subslots_{id(self)}"] = _bound_dict(value)


_SubSlotsRef = SubSlots  # internal alias (the metaclass collects by this type)


def _bound_dict(binding) -> dict:
    """The concrete dict from a ``sub_slots`` binding (a ``SubSlots.from_dict`` or a
    bare dict)."""
    if isinstance(binding, SubSlots):
        return dict(binding.bound or {})
    return dict(binding)


def arg(default=_NODEFAULT) -> Any:
    """A value-input slot, as a *unique* slot node carrying its default. ``default``
    is what it resolves to when left unbound: usually ``Input`` (the global input),
    but also a parameter / tensor (bound in as a source value node) or any handle.
    With no default the slot is required. Binding a slot substitutes only that
    slot's node -- never the global ``Input`` or another slot. (The runtime marker is a
    plain interned ``ValueRef``; ``ArgSlot`` is the *type-only* descriptor the ``Arg``
    annotation resolves to -- see ``ArgSlot``.)"""
    return ref(_ArgSlot(default), ())


def op_slot[M](default=_NODEFAULT) -> Slot[M]:
    """A producer slot (a ``cl.Module`` / ``Op`` / callable), optionally defaulting to
    a concrete producer when left unbound. Returns a :class:`Slot` -- in the body
    ``enc(x)`` applies it and ``enc.get(lambda m: m.weight)`` reads a param; construct
    with ``Spec(enc=module_or_fn)``. The same descriptor as ``module_slot()`` -- an op
    and a leaf module are one idea (a callable you also read attributes off)."""
    return Slot(_SlotMarker(default))


def sub_slot[M]() -> Slot[M]:
    """A *sub-model* slot: a whole sub-graph (type ``M``, a :class:`GraphSpec`)
    plugged into a graph as a :class:`Slot` descriptor. ``m(x)`` plugs it in with its
    ``Input`` rebound to ``x``, and ``m(x).recon`` references the sub-graph's ``recon``
    handle (no ``.get`` -- a sub-graph exposes its outputs as named handles). The same
    descriptor as ``op_slot()`` / ``module_slot()`` -- the difference is only the body
    face: a leaf has one forward value, a sub-graph has named handles."""
    return Slot(_SlotMarker())


def sub_slots() -> Any:
    """A *variadic* sub-model slot: bound to a ``dict`` of sub-graphs at construction
    (bare or via ``SubSlots.from_dict``); fold over them declaratively in the class
    body with ``.reduce(per_item, combine)`` (e.g. a weighted loss sum over a
    dynamic set)."""
    return SubSlots()


def module_slot[M]() -> Slot[M]:
    """A *module* slot: a leaf ``cl.Module`` plugged into a graph as a :class:`Slot`
    descriptor. Declare it ``norm: Slot[AffineNormalizer] = module_slot()`` and
    construct with ``Spec(norm=module)`` (the descriptor ``__set__`` takes the module).
    In the body ``norm(x)`` is its forward and ``norm.get(lambda n: n.scale)`` reads a
    param/buffer as a typed graph ref; on the constructed instance ``self.norm`` is the
    bound module (so ``self.norm.primed`` type-checks). The same descriptor as
    ``op_slot()`` -- a leaf module is just a producer you also read buffers off."""
    return Slot(_SlotMarker())


def handle(*, init: bool = False) -> Any:
    """Declare a GraphSpec *output-handle contract*: a typed instance attribute that
    subclasses wire as a body handle (``recon = decoder(acts)``) and the metaclass
    binds -- *not* a construction field. A ``dataclass_transform`` field-specifier
    with ``init=False``, so the handle is typed on the base (and satisfies a structural
    protocol) without being demanded at construction *and* without forcing an
    annotation on every subclass handle. Runtime-inert: the base's ``recon: ValueRef =
    handle()`` is a sentinel the metaclass skips, and a subclass's real handle node
    (``recon = decoder(acts)``) shadows it. (The ``init`` parameter exists only so a
    type checker reads the field as ``init=False``; callers write ``handle()``.)"""
    return _HANDLE_FIELD


@define
class _Template:
    output: ValueRef | None  # None == abstract (cannot be constructed)
    value_slots: dict[str, ValueRef]
    slots: dict[str, Slot]  # op + module + sub-graph slots (one `Slot` descriptor)
    multi_slots: dict[str, _SubSlotsRef]
    handles: dict[str, ValueRef]
    partials: dict[str, _PartialGraph]


@define(frozen=True)
class _SlotSpec:
    name: str
    kind: str
    marker: object
    default: object = _NODEFAULT
    required: bool = True


@define
class _ResolvedSlots:
    subs: Subs
    open_slots: dict[str, _SlotSpec]
    bound_slots: dict[str, object]  # slot -> raw binding (the typed `self.x` / child)
    multi_subgraphs: dict[str, dict[str, GraphSpec]]


@define
class _PartialGraph[T]:
    """A partially-bound ``GraphSpec`` (some slots closed, some still open), generic
    over the spec type ``T`` it is a partial *of* -- so ``Denormalizer.partial(...)``
    is a ``_PartialGraph[Denormalizer]`` and ``.source`` is ``type[Denormalizer]``."""

    source: type[T]
    handles: dict[str, ValueRef]
    out: ValueRef | None
    open_slots: dict[str, _SlotSpec]

    def __getattr__(self, name: str) -> ValueRef[torch.Tensor]:
        try:
            return self.handles[name]
        except KeyError as e:
            raise AttributeError(name) from e

    @property
    def output(self) -> ValueRef[torch.Tensor]:
        if "output" in self.handles:
            return self.handles["output"]
        if self.out is not None:
            return self.out
        raise AttributeError("output")

    def rebind(self, subs: Subs) -> _PartialGraph[T]:
        # rebind the materialized handles AND any still-open slot's ValueRef default,
        # so an omitted default (e.g. arg(default=Input)) follows the same structural
        # rebase as the rest of the partial rather than resolving to the stale node.
        def _rebind_slot(s: _SlotSpec) -> _SlotSpec:
            if isinstance(s.default, ValueRef):
                d = bind(s.default, subs)
                return _SlotSpec(s.name, s.kind, s.marker, d, s.required)
            return s

        return _PartialGraph(
            self.source,
            {nm: bind(h, subs) for nm, h in self.handles.items()},
            bind(self.out, subs) if self.out is not None else None,
            {nm: _rebind_slot(s) for nm, s in self.open_slots.items()},
        )

    def complete(self, **bindings) -> GraphSpec:
        resolved = _resolve_slots(
            self.source.__name__, self.open_slots, bindings, require_all=True
        )
        if self.out is None:
            raise TypeError(f"{self.source.__name__} has no `output` to complete")
        handles = {nm: bind(h, resolved.subs) for nm, h in self.handles.items()}
        out = bind(self.out, resolved.subs)
        handles.setdefault("output", out)
        return GraphSpec.of(**handles)

    def __call__(self, value) -> GraphSpec:
        if len(self.open_slots) != 1:
            raise TypeError(
                f"{self.source.__name__} partial has {len(self.open_slots)} "
                "open slots; positional completion requires exactly one"
            )
        (name,) = self.open_slots
        return self.complete(**{name: value})


def _merge_templates(templates: list[_Template]) -> _Template:
    value_slots, slots, multi_slots, handles, partials = ({}, {}, {}, {}, {})
    output = None
    for t in templates:
        value_slots.update(t.value_slots)
        slots.update(t.slots)
        multi_slots.update(t.multi_slots)
        handles.update(t.handles)
        partials.update(t.partials)
        if t.output is not None:
            output = t.output
    return _Template(output, value_slots, slots, multi_slots, handles, partials)


def _iter_slots(t: _Template):
    for nm, slot in t.slots.items():
        marker = slot._marker
        default = marker.default
        yield _SlotSpec(nm, "slot", marker, default, default is _NODEFAULT)
    for nm, vref in t.value_slots.items():
        marker = vref.producer
        default = marker.default if isinstance(marker, _ArgSlot) else _NODEFAULT
        yield _SlotSpec(nm, "value", vref, default, default is _NODEFAULT)
    for nm, msref in t.multi_slots.items():
        yield _SlotSpec(nm, "multi", msref.slots)


def _slot_map(t: _Template) -> dict[str, _SlotSpec]:
    return {slot.name: slot for slot in _iter_slots(t)}


def _slot_substitution(slot: _SlotSpec, binding) -> tuple[object, object]:
    if slot.kind == "slot":
        if isinstance(binding, GraphSpec):  # a sub-graph: keep raw for handle splice
            return slot.marker, binding
        producer = (  # a module / op / callable: normalize to a bound producer
            binding.producer if isinstance(binding, OpRef) else op(binding).producer
        )
        return slot.marker, producer
    if slot.kind == "value":
        value = binding if isinstance(binding, ValueRef) else ref(binding, ())
        return slot.marker, value
    if slot.kind == "multi":
        return slot.marker, _bound_dict(binding)
    raise AssertionError(f"unknown slot kind {slot.kind!r}")


def _slot_default_substitution(slot: _SlotSpec) -> tuple[object, object]:
    if slot.kind == "slot":
        return slot.marker, op(slot.default).producer
    if slot.kind == "value":
        default = slot.default
        value = default if isinstance(default, ValueRef) else ref(default, ())
        return slot.marker, value
    raise AssertionError(f"{slot.kind!r} slots do not have defaults")


def _missing_required_message(source_name: str, slot: _SlotSpec) -> str:
    if slot.kind == "value":
        return f"{source_name} missing required value slot {slot.name!r}"
    if slot.kind == "multi":
        return f"{source_name} missing required sub-models {slot.name!r}"
    return f"{source_name} missing required slot {slot.name!r}"


def _resolve_slots(
    source_name: str,
    slots: dict[str, _SlotSpec],
    bindings: dict[str, Any],
    *,
    require_all: bool,
) -> _ResolvedSlots:
    for nm in bindings:
        if nm not in slots:
            raise TypeError(f"{source_name} has no open slot {nm!r}")
    subs: Subs = {}
    open_slots: dict[str, _SlotSpec] = {}
    bound_slots: dict[str, object] = {}
    multi_subgraphs: dict[str, dict] = {}
    for nm, slot in slots.items():
        if nm in bindings:
            key, value = _slot_substitution(slot, bindings[nm])
            subs[key] = value
            if slot.kind == "slot":  # the typed `self.x` (module/op or sub-graph child)
                bound_slots[nm] = bindings[nm]
            elif slot.kind == "multi":
                multi_subgraphs[nm] = value
            continue
        if not require_all:
            open_slots[nm] = slot
            continue
        if slot.default is not _NODEFAULT:
            key, value = _slot_default_substitution(slot)
            subs[key] = value
            continue
        if slot.required:
            raise TypeError(_missing_required_message(source_name, slot))
    return _ResolvedSlots(subs, open_slots, bound_slots, multi_subgraphs)


def _resolve_template_bindings(
    cls: type,
    t: _Template,
    bindings: dict[str, Any],
    *,
    require_all: bool,
) -> _ResolvedSlots:
    slots = _slot_map(t)
    for nm in bindings:
        if nm not in slots:
            raise TypeError(f"{cls.__name__} has no slot {nm!r}")
    return _resolve_slots(cls.__name__, slots, bindings, require_all=require_all)


def _is_subgraph_proxy(node) -> bool:
    """A bound handle whose producer is a whole ``GraphSpec`` is a sub-graph
    application proxy (``nx = normalizer(x)``), not a value -- its ``.attr`` handles
    are the real nodes. Excluded from a spec's handles, uniformly across construction
    paths (``GraphSpec(...)`` and ``GraphSpec.of(...)`` / ``partial().complete()``)."""
    return isinstance(getattr(node, "producer", None), GraphSpec)


@dataclass_transform(
    field_specifiers=(arg, op_slot, sub_slot, sub_slots, module_slot, handle)
)
class _GraphMeta(type):
    def __new__(mcs, name, bases, ns, **kw):
        # keep the body refs as class attributes (the *templates*): a type
        # checker reads them off the class to give `spec.acts: ValueRef` and to
        # flag typos. Each instance shadows them with its bound handles.
        cls = super().__new__(mcs, name, bases, dict(ns))
        if not any(isinstance(b, _GraphMeta) for b in bases):
            return cls  # the GraphSpec base itself
        own_value, own_slot, own_multi = {}, {}, {}
        for slot_name in ns.get("__annotations__", {}):
            if slot_name == "output" or slot_name.startswith("_"):
                continue
            v = ns.get(slot_name)
            if isinstance(v, Slot):  # op + module + sub-graph slots (one descriptor)
                own_slot[slot_name] = v
            elif isinstance(v, _SubSlotsRef):
                own_multi[slot_name] = v
            elif isinstance(v, ValueRef):
                own_value[slot_name] = v
        # `_AppliedSlot` (a sub-graph application proxy, e.g. `nx = normalizer(x)`) is a
        # ValueRef, so it lands in own_handles; a *leaf* slot's `slot(x)` is a real
        # forward value, but a *sub-graph* slot's `slot(x)` is only a carrier for
        # `.attr` handles -- the latter is filtered out at construction (its bound node
        # has a GraphSpec producer; a whole sub-graph isn't a single value).
        own_handles = {k: v for k, v in ns.items() if isinstance(v, ValueRef)}
        own_partials = {k: v for k, v in ns.items() if isinstance(v, _PartialGraph)}
        own_out = ns.get("output")
        own_output = own_out if isinstance(own_out, ValueRef) else None

        spec_bases = [
            b
            for b in bases
            if isinstance(b, _GraphMeta) and getattr(b, "_template", None)
        ]
        if spec_bases:
            base = _merge_templates([b._template for b in spec_bases])
            # an overridden slot's handle references the *base* marker; re-point
            # those references to this class's marker (so e.g. a new default wins)
            subs: Subs = {}
            for nm, slot in own_slot.items():  # overridden slot: re-point its marker
                if nm in base.slots:
                    subs[base.slots[nm]._marker] = slot._marker
            for nm, vref in own_value.items():
                if nm in base.value_slots:
                    subs[base.value_slots[nm]] = vref
            handles = {k: bind(h, subs) for k, h in base.handles.items()}
            handles.update(own_handles)
            partials = {k: p.rebind(subs) for k, p in base.partials.items()}
            partials.update(own_partials)
            output = (
                own_output
                if own_output is not None
                else (bind(base.output, subs) if base.output is not None else None)
            )
            value_slots = {**base.value_slots, **own_value}
            slots = {**base.slots, **own_slot}
            multi_slots = {**base.multi_slots, **own_multi}
        else:
            value_slots, slots = own_value, own_slot
            multi_slots = own_multi
            handles, output = own_handles, own_output
            partials = own_partials

        cls._template = _Template(
            output, value_slots, slots, multi_slots, handles, partials
        )
        return cls

    def __call__(cls, **bindings):
        t: _Template = cls._template
        if t.output is None:
            raise TypeError(f"{cls.__name__} is abstract (no `output`)")
        resolved = _resolve_template_bindings(
            cls, t, bindings, require_all=True
        )
        obj = object.__new__(cls)
        # bound handles as instance attributes -- they shadow the class templates,
        # so `spec.acts` is the inlined node at runtime and typed at check time. Drop
        # any handle that bound to a GraphSpec producer: that is a sub-graph
        # application proxy (`nx = normalizer(x)`), not a value -- its `.attr` handles
        # are separate nodes and survive.
        handle_refs = {}
        for nm, h in t.handles.items():
            bound = bind(h, resolved.subs)
            if _is_subgraph_proxy(bound):
                continue
            handle_refs[nm] = bound
            setattr(obj, nm, bound)
        partial_refs = {nm: p.rebind(resolved.subs) for nm, p in t.partials.items()}
        for nm, partial in partial_refs.items():
            setattr(obj, nm, partial)
        # bound components as typed instance attrs (`self.norm` is the module, `self.
        # model` is the sub-graph -- a navigable child) -- via the `Slot` descriptor.
        for nm, comp in resolved.bound_slots.items():
            setattr(obj, nm, comp)
        for nm, children in resolved.multi_subgraphs.items():
            setattr(obj, nm, children)
        obj._setup(handle_refs, bind(t.output, resolved.subs), partial_refs)
        return obj


@dataclass_transform(
    field_specifiers=(arg, op_slot, sub_slot, sub_slots, module_slot, handle)
)
class GraphSpec(metaclass=_GraphMeta):
    """A graph: the authoring surface *and* the runnable context.

    Subclasses declare slots as annotated fields and wire the body; constructing
    binds the slots and exposes the bound handles as typed instance attributes
    (``g.acts``; a typo errors). The instance is then **the graph**: run it with
    ``g.forward_values(x)`` (a fused, memoized, multi-output pass), pull bubbled-up
    sites with ``g.collect(SiteType)``, and register its parameters/buffers onto
    an owning ``nn.Module`` with ``g.register_into(module)``. Bound sub-graphs are
    kept as navigable children (``g.norm_model.y_pred_norm``), not flattened.

    A ref means something only relative to its graph -- so always reach handles
    through the graph, never pass them around bare.
    """

    def _setup(
        self,
        handle_refs: dict[str, ValueRef],
        out: ValueRef,
        partials: dict[str, _PartialGraph] | None = None,
    ) -> None:
        self._handle_refs = handle_refs
        self._out = out
        self._partials = partials or {}
        self._names = None  # lazy name map over the handle cones

    @classmethod
    def partial(cls, **bindings) -> _PartialGraph[Self]:
        """Bind a subset of this spec's slots, leaving the rest as open markers.

        The returned partial exposes the same handle names, can be structurally
        rebound with ``.rebind(...)``, and can be completed once every required
        open slot has a binding.
        """
        t: _Template = cls._template
        resolved = _resolve_template_bindings(
            cls, t, bindings, require_all=False
        )
        return _PartialGraph(
            cls,
            {nm: bind(h, resolved.subs) for nm, h in t.handles.items()},
            bind(t.output, resolved.subs) if t.output is not None else None,
            resolved.open_slots,
        )

    @classmethod
    def of(cls, output: ValueRef | None = None, **handles: ValueRef) -> GraphSpec:
        """Build a graph dynamically from named handles -- for aggregates assembled
        at runtime (e.g. the trainable's whole computation), where a class body
        can't enumerate a runtime-sized set. Handles are typed-attribute-accessed
        (``g.total_loss``); ``output`` (or a handle named ``output``) is the
        default ``forward`` output."""
        obj = object.__new__(GraphSpec)
        out = output if output is not None else handles.get("output")
        if out is None:
            raise TypeError("GraphSpec.of needs an `output` handle")
        # expose `output` as a handle too (so `g.output` works like a normal spec,
        # not just `g.out`) -- whether it came in as the param or in **handles. Drop
        # sub-application proxies, same as `GraphSpec(...)` (so `partial().complete()`
        # and direct construction agree).
        handles = {
            "output": out,
            **{nm: h for nm, h in handles.items() if not _is_subgraph_proxy(h)},
        }
        for nm, h in handles.items():
            setattr(obj, nm, h)
        obj._setup(handles, out)
        return obj

    # -- the graph as context ---------------------------------------------------
    @property
    def out(self) -> ValueRef:
        return self._out

    def __repr__(self) -> str:
        n = len(_toposort(*self._targets()))
        return f"<{type(self).__name__}: {', '.join(self._handle_refs)} | {n} nodes>"

    def render(self) -> str:
        """A flat, deduped view of the whole graph: every node in the cone of all
        handles printed *once* (so a shared node -- the encoder, the normalize op
        -- appears a single time), each tagged with the handle name(s) pointing at
        it. The execution view (D-02), and the picture that makes "it all fuses to
        one pass" visible -- the deduped DAG."""
        order = _toposort(*self._targets())
        aliases: dict[int, list[str]] = {}
        for nm, h in self._handle_refs.items():
            aliases.setdefault(id(h), []).append(nm)
        name: dict[int, str] = {}
        for i, n in enumerate(order):
            if n is Input:
                name[id(n)] = "input"
            elif id(n) in aliases:
                name[id(n)] = aliases[id(n)][0]
            else:
                name[id(n)] = f"n{i}"
        width = max((len(v) for v in name.values()), default=4)
        lines = [f"{type(self).__name__} ({len(order)} nodes):"]
        for n in order:
            ins = ", ".join(name[id(i)] for i in n.inputs)
            arrow = f"  <- {ins}" if ins else ""
            primary = name[id(n)]
            others = [a for a in aliases.get(id(n), []) if a != primary]
            tag = f"   [{', '.join(others)}]" if others else ""
            label = _producer_label(n.producer)
            lines.append(f"  {primary:<{width}} = {label}{arrow}{tag}")
        return "\n".join(lines)

    @property
    def names(self) -> dict[ValueRef, str]:
        if self._names is None:
            self._names = _name_map(_toposort(*self._targets()))
        return self._names

    def _targets(self) -> tuple[ValueRef, ...]:
        return (self._out, *self._handle_refs.values())

    def iter_nodes(self) -> Iterator[ValueRef]:
        """Iterate this graph's nodes in dependency order (inputs first) -- every node
        in the cone of all handles, deduped (a shared node appears once). The "walk my
        cone" op behind ``render()`` / ``collect()`` / ``register_into()``, exposed for
        introspection and filtering."""
        return iter(_toposort(*self._targets()))

    def forward_values(
        self, x, *, cache, training: bool = True, **kwargs
    ) -> _SaveSession:
        """One fused pass: ``save(handle)`` realizes a handle (memoized across
        saves), ``save.after(h, fn)`` defers a side-effect to block exit. Pass
        ``training=False`` (an eval pass) to skip training-only deferrals -- e.g. so
        an EMA tracker doesn't advance during validation."""
        return _SaveSession(self, x, cache, kwargs, training=training)

    def collect[S](self, site_type: type[S]) -> list[ValueRef[Any, S]]:
        """Bubble up: nodes in this graph's cone whose producer is ``site_type``
        (typed ``ValueRef[Any, S]``, so ``node.producer`` is an ``S``)."""
        return collect(list(self._targets()), site_type)

    def register_into(self, module) -> GraphSpec:
        """Register this graph's leaf producers (params/buffers/modules) onto an
        owning ``nn.Module`` so ``.parameters()`` / ``.to()`` / ``state_dict`` see
        them. The graph runs off live refs; this is just nn.Module bookkeeping."""
        _register_producers(module, _toposort(*self._targets()), self.names)
        return module

    def forward(self, x, *, cache, **kwargs):
        with self.forward_values(x, cache=cache, **kwargs) as save:
            return save(self._out)

    def apply_to(self, x: ValueRef) -> Self:
        """Apply this (bound) graph to a new input: a view whose handles read
        ``x`` instead of ``Input`` -- the standalone form of a sub-model slot's
        ``m(x)``."""
        clone = object.__new__(type(self))
        refs = {nm: bind(h, {Input: x}) for nm, h in self._handle_refs.items()}
        for nm, h in refs.items():
            setattr(clone, nm, h)
        partials = {nm: p.rebind({Input: x}) for nm, p in self._partials.items()}
        for nm, partial in partials.items():
            setattr(clone, nm, partial)
        clone._setup(refs, bind(self._out, {Input: x}), partials)
        return clone

    __call__ = apply_to

    def produce(self, ref: ValueRef) -> Graph:
        """A Graph targeting one handle -- runs only that handle's cone."""
        return ref.build()

    def __add__(self, o) -> ValueRef[torch.Tensor]:
        return self._out + (o.out if isinstance(o, GraphSpec) else o)

    def __mul__(self, o) -> ValueRef[torch.Tensor]:
        return self._out * (o.out if isinstance(o, GraphSpec) else o)

    def __sub__(self, o) -> ValueRef[torch.Tensor]:
        return self._out - (o.out if isinstance(o, GraphSpec) else o)

    def __truediv__(self, o) -> ValueRef[torch.Tensor]:
        return self._out / (o.out if isinstance(o, GraphSpec) else o)
