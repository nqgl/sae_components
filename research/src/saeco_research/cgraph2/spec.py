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

from collections.abc import Callable
from typing import Any, Self, dataclass_transform

from attrs import define, field

from saeco_research.cgraph2.core import (
    Graph,
    Input,
    OpRef,
    ValueRef,
    _producer_label,
    _register_producers,
    _SaveSession,
    _toposort,
    bind,
    collect,
    op,
    ref,
)
from saeco_research.cgraph2.core import (
    _names as _name_map,
)

# Annotation aliases matching the slot vocabulary.
Arg = ValueRef
Op = OpRef

_NODEFAULT = object()


@define(eq=False)
class _OpSlot:
    """Marker producer for an unbound producer slot; ``default`` (if set) is the
    producer used when the slot is left unbound at construction."""

    default: object = _NODEFAULT


@define(eq=False)
class _ArgSlot:
    """Marker producer for a value slot. ``default`` is what the slot resolves to
    when left unbound (``_NODEFAULT`` == required). Each ``arg()`` is a *unique*
    marker, so binding one slot never disturbs another (or the global ``Input``)."""

    default: object = _NODEFAULT


@define(eq=False)
class _SubSlot:
    """Marker for a sub-model slot, bound to a sub-graph bundle at construction."""


@define(eq=False)
class _SubGet:
    """Producer of a ``sub(x).attr`` node. At bind time it splices in the bound
    sub-graph's ``attr`` handle with that handle's ``Input`` rebound to ``x``."""

    slot: _SubSlot
    attr: str

    def __graph_rebind__(self, inputs: tuple, subs: dict) -> ValueRef:
        bound = subs.get(self.slot, None)
        if bound is None:  # slot not in subs: keep the node (re-interned inputs)
            return ref(self, inputs)
        if isinstance(bound, _SubSlot):  # inheritance re-point: swap the marker
            return ref(_SubGet(bound, self.attr), inputs)
        (x_node,) = inputs  # construction: splice the bound sub-graph
        return bind(getattr(bound, self.attr), {Input: x_node})


class _AppliedSub:
    """``sub_model(x)`` at template time; ``.attr`` yields a deferred sub-get."""

    def __init__(self, slot: _SubSlot, x: ValueRef):
        self._slot = slot
        self._x = x

    def __getattr__(self, attr: str) -> ValueRef:
        if attr.startswith("_"):
            raise AttributeError(attr)
        return ref(_SubGet(self._slot, attr), (self._x,))


@define(eq=False)
class _SubSlotRef:
    """What ``sub_slot()`` returns; callable to apply the sub-model to an input."""

    slot: _SubSlot

    def __call__(self, x: ValueRef) -> Any:
        return _AppliedSub(self.slot, x)


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

    def __graph_rebind__(self, inputs: tuple, subs: dict) -> ValueRef:
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
    slot's node -- never the global ``Input`` or another slot."""
    return ref(_ArgSlot(default), ())


def op_slot(default=_NODEFAULT) -> Any:
    """A producer slot, optionally defaulting to a concrete producer."""
    return OpRef(_OpSlot(default))


def sub_slot() -> Any:
    """A sub-model slot: bound to a sub-graph bundle, applied via ``m(x)``."""
    return _SubSlotRef(_SubSlot())


def sub_slots() -> Any:
    """A *variadic* sub-model slot: bound to a ``dict`` of sub-graphs at construction
    (bare or via ``SubSlots.from_dict``); fold over them declaratively in the class
    body with ``.reduce(per_item, combine)`` (e.g. a weighted loss sum over a
    dynamic set)."""
    return SubSlots()


@define
class _Template:
    output: ValueRef | None  # None == abstract (cannot be constructed)
    value_slots: dict[str, ValueRef]
    op_slots: dict[str, OpRef]
    sub_slots: dict[str, _SubSlotRef]
    multi_slots: dict[str, _SubSlotsRef]
    handles: dict[str, ValueRef]


def _merge_templates(templates: list[_Template]) -> _Template:
    value_slots, op_slots, sub_slots, multi_slots, handles = {}, {}, {}, {}, {}
    output = None
    for t in templates:
        value_slots.update(t.value_slots)
        op_slots.update(t.op_slots)
        sub_slots.update(t.sub_slots)
        multi_slots.update(t.multi_slots)
        handles.update(t.handles)
        if t.output is not None:
            output = t.output
    return _Template(output, value_slots, op_slots, sub_slots, multi_slots, handles)


@dataclass_transform(field_specifiers=(arg, op_slot, sub_slot, sub_slots))
class _GraphMeta(type):
    def __new__(mcs, name, bases, ns, **kw):
        # keep the body refs as class attributes (the *templates*): a type
        # checker reads them off the class to give `spec.acts: ValueRef` and to
        # flag typos. Each instance shadows them with its bound handles.
        cls = super().__new__(mcs, name, bases, dict(ns))
        if not any(isinstance(b, _GraphMeta) for b in bases):
            return cls  # the GraphSpec base itself
        own_value, own_op, own_sub, own_multi = {}, {}, {}, {}
        for slot_name in ns.get("__annotations__", {}):
            if slot_name == "output" or slot_name.startswith("_"):
                continue
            v = ns.get(slot_name)
            if isinstance(v, OpRef):
                own_op[slot_name] = v
            elif isinstance(v, _SubSlotRef):
                own_sub[slot_name] = v
            elif isinstance(v, _SubSlotsRef):
                own_multi[slot_name] = v
            elif isinstance(v, ValueRef):
                own_value[slot_name] = v
        own_handles = {k: v for k, v in ns.items() if isinstance(v, ValueRef)}
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
            subs: dict = {}
            for nm, oref in own_op.items():
                if nm in base.op_slots:
                    subs[base.op_slots[nm].producer] = oref.producer
            for nm, vref in own_value.items():
                if nm in base.value_slots:
                    subs[base.value_slots[nm]] = vref
            for nm, sref in own_sub.items():  # overridden sub-model slot
                if nm in base.sub_slots:
                    subs[base.sub_slots[nm].slot] = sref.slot
            handles = {k: bind(h, subs) for k, h in base.handles.items()}
            handles.update(own_handles)
            output = (
                own_output
                if own_output is not None
                else (bind(base.output, subs) if base.output is not None else None)
            )
            value_slots = {**base.value_slots, **own_value}
            op_slots = {**base.op_slots, **own_op}
            sub_slots = {**base.sub_slots, **own_sub}
            multi_slots = {**base.multi_slots, **own_multi}
        else:
            value_slots, op_slots, sub_slots = own_value, own_op, own_sub
            multi_slots = own_multi
            handles, output = own_handles, own_output

        cls._template = _Template(
            output, value_slots, op_slots, sub_slots, multi_slots, handles
        )
        return cls

    def __call__(cls, **bindings):
        t: _Template = cls._template
        if t.output is None:
            raise TypeError(f"{cls.__name__} is abstract (no `output`)")
        known = (
            set(t.value_slots) | set(t.op_slots) | set(t.sub_slots) | set(t.multi_slots)
        )
        for nm in bindings:
            if nm not in known:
                raise TypeError(f"{cls.__name__} has no slot {nm!r}")
        subs: dict = {}
        for nm, oref in t.op_slots.items():
            marker = oref.producer
            if nm in bindings:
                b = bindings[nm]
                subs[marker] = b.producer if isinstance(b, OpRef) else op(b).producer
            elif isinstance(marker, _OpSlot) and marker.default is not _NODEFAULT:
                subs[marker] = op(marker.default).producer
            elif isinstance(marker, _OpSlot):
                raise TypeError(f"{cls.__name__} missing required slot {nm!r}")
        for nm, vref in t.value_slots.items():
            marker = vref.producer  # _ArgSlot
            if nm in bindings:
                b = bindings[nm]
                # bind to a handle/Input directly, or wrap a raw param/tensor.
                subs[vref] = b if isinstance(b, ValueRef) else ref(b, ())
            elif marker.default is _NODEFAULT:
                raise TypeError(f"{cls.__name__} missing required value slot {nm!r}")
            else:  # resolve the unbound slot to its default (Input / param / handle)
                d = marker.default
                subs[vref] = d if isinstance(d, ValueRef) else ref(d, ())
        for nm, sref in t.sub_slots.items():
            if nm in bindings:
                subs[sref.slot] = bindings[nm]  # the bound sub-graph bundle
            else:
                raise TypeError(f"{cls.__name__} missing required sub-model {nm!r}")
        for nm, msref in t.multi_slots.items():
            if nm in bindings:
                subs[msref.slots] = _bound_dict(bindings[nm])  # the bound sub-graphs
            else:
                raise TypeError(f"{cls.__name__} missing required sub-models {nm!r}")
        obj = object.__new__(cls)
        # bound handles as instance attributes -- they shadow the class templates,
        # so `spec.acts` is the inlined node at runtime and typed at check time.
        handle_refs = {nm: bind(h, subs) for nm, h in t.handles.items()}
        for nm, h in handle_refs.items():
            setattr(obj, nm, h)
        # keep bound sub-graphs as navigable children (structure, not flattened):
        # spec.norm_model.y_pred_norm works (D-02).
        subgraphs = {nm: bindings[nm] for nm in t.sub_slots if nm in bindings}
        for nm, child in subgraphs.items():
            setattr(obj, nm, child)
        for nm in t.multi_slots:  # variadic children: a navigable dict of sub-graphs
            if nm in bindings:
                setattr(obj, nm, _bound_dict(bindings[nm]))
        obj._setup(handle_refs, bind(t.output, subs), subgraphs)
        return obj


@dataclass_transform(field_specifiers=(arg, op_slot, sub_slot, sub_slots))
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
        subgraphs: dict[str, GraphSpec],
    ) -> None:
        self._handle_refs = handle_refs
        self._out = out
        self._subgraphs = subgraphs
        self._names = None  # lazy name map over the handle cones

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
        for nm, h in handles.items():
            setattr(obj, nm, h)
        obj._setup(dict(handles), out, {})
        return obj

    # -- the graph as context ---------------------------------------------------
    @property
    def out(self) -> ValueRef:
        return self._out

    def __repr__(self) -> str:
        n = len(_toposort(*self._targets()))
        return f"<{type(self).__name__}: {', '.join(self._handle_refs)} | {n} nodes>"

    def describe(self, indent: int = 0) -> str:
        """A readable tree of this graph's handles and its (kept) sub-graph
        children -- ``print(g.describe())`` shows the nested structure (D-02)."""
        pad = "  " * indent
        lines = [f"{pad}{type(self).__name__}  handles: {list(self._handle_refs)}"]
        for nm, child in self._subgraphs.items():
            if isinstance(child, GraphSpec):
                lines.append(f"{pad}  {nm}:")
                lines.append(child.describe(indent + 2))
            else:
                lines.append(f"{pad}  {nm}: {child!r}")
        return "\n".join(lines)

    def render(self) -> str:
        """A flat, deduped view of the whole graph: every node in the cone of all
        handles printed *once* (so a shared node -- the encoder, the normalize op
        -- appears a single time), each tagged with the handle name(s) pointing at
        it. The execution view (D-02), and the picture that makes "it all fuses to
        one pass" visible. ``describe()`` is the complementary nested-structure
        view; this is the deduped DAG."""
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
    def names(self) -> dict:
        if self._names is None:
            self._names = _name_map(_toposort(*self._targets()))
        return self._names

    def _targets(self) -> tuple[ValueRef, ...]:
        return (self._out, *self._handle_refs.values())

    def forward_values(self, x, *, cache, **kwargs) -> _SaveSession:
        """One fused pass: ``save(handle)`` realizes a handle (memoized across
        saves), ``save.after(h, fn)`` defers a side-effect to block exit."""
        return _SaveSession(self, x, cache, kwargs)

    def collect[S](self, site_type: type[S]) -> list[ValueRef[S]]:
        """Bubble up: nodes in this graph's cone whose producer is ``site_type``
        (typed ``ValueRef[S]``, so ``node.producer`` is an ``S``)."""
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
        clone._setup(refs, bind(self._out, {Input: x}), {})
        return clone

    __call__ = apply_to

    def produce(self, ref: ValueRef) -> Graph:
        """A Graph targeting one handle -- runs only that handle's cone."""
        return ref.build()

    def __add__(self, o) -> ValueRef:
        return self._out + (o.out if isinstance(o, GraphSpec) else o)

    def __mul__(self, o) -> ValueRef:
        return self._out * (o.out if isinstance(o, GraphSpec) else o)

    def __sub__(self, o) -> ValueRef:
        return self._out - (o.out if isinstance(o, GraphSpec) else o)

    def __truediv__(self, o) -> ValueRef:
        return self._out / (o.out if isinstance(o, GraphSpec) else o)
