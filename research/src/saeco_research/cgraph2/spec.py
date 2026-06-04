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

from typing import Any, Self, dataclass_transform

from attrs import define

from saeco_research.cgraph2.core import (
    Graph,
    Input,
    OpRef,
    ValueRef,
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
    """Marker producer for a *required* value slot (``arg()`` with no default)."""


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
        if self.slot not in subs:  # not yet bound (e.g. inheritance re-point)
            return ref(self, inputs)
        (x_node,) = inputs
        return bind(getattr(subs[self.slot], self.attr), {Input: x_node})


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


def arg(default=_NODEFAULT) -> Any:
    """A value-input slot. ``default`` is what it resolves to when left unbound:
    usually ``Input`` (the global input), but also a parameter / tensor (bound in
    as a source value node) or any handle. With no default the slot is required."""
    if default is _NODEFAULT:
        return ref(_ArgSlot(), ())  # required
    if isinstance(default, ValueRef):
        return default  # Input, a handle, etc.
    return ref(default, ())  # a param / tensor -> a source value node


def op_slot(default=_NODEFAULT) -> Any:
    """A producer slot, optionally defaulting to a concrete producer."""
    return OpRef(_OpSlot(default))


def sub_slot() -> Any:
    """A sub-model slot: bound to a sub-graph bundle, applied via ``m(x)``."""
    return _SubSlotRef(_SubSlot())


@define
class _Template:
    output: ValueRef | None  # None == abstract (cannot be constructed)
    value_slots: dict[str, ValueRef]
    op_slots: dict[str, OpRef]
    sub_slots: dict[str, _SubSlotRef]
    handles: dict[str, ValueRef]


def _merge_templates(templates: list[_Template]) -> _Template:
    value_slots, op_slots, sub_slots, handles = {}, {}, {}, {}
    output = None
    for t in templates:
        value_slots.update(t.value_slots)
        op_slots.update(t.op_slots)
        sub_slots.update(t.sub_slots)
        handles.update(t.handles)
        if t.output is not None:
            output = t.output
    return _Template(output, value_slots, op_slots, sub_slots, handles)


@dataclass_transform(field_specifiers=(arg, op_slot, sub_slot))
class _GraphMeta(type):
    def __new__(mcs, name, bases, ns, **kw):
        # keep the body refs as class attributes (the *templates*): a type
        # checker reads them off the class to give `spec.acts: ValueRef` and to
        # flag typos. Each instance shadows them with its bound handles.
        cls = super().__new__(mcs, name, bases, dict(ns))
        if not any(isinstance(b, _GraphMeta) for b in bases):
            return cls  # the GraphSpec base itself
        own_value, own_op, own_sub = {}, {}, {}
        for slot_name in ns.get("__annotations__", {}):
            if slot_name == "output" or slot_name.startswith("_"):
                continue
            v = ns.get(slot_name)
            if isinstance(v, OpRef):
                own_op[slot_name] = v
            elif isinstance(v, _SubSlotRef):
                own_sub[slot_name] = v
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
        else:
            value_slots, op_slots, sub_slots = own_value, own_op, own_sub
            handles, output = own_handles, own_output

        cls._template = _Template(output, value_slots, op_slots, sub_slots, handles)
        return cls

    def __call__(cls, **bindings):
        t: _Template = cls._template
        if t.output is None:
            raise TypeError(f"{cls.__name__} is abstract (no `output`)")
        known = set(t.value_slots) | set(t.op_slots) | set(t.sub_slots)
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
            if nm in bindings:
                b = bindings[nm]
                # bind to a handle/Input directly, or wrap a raw param/tensor.
                subs[vref] = b if isinstance(b, ValueRef) else ref(b, ())
            elif isinstance(vref.producer, _ArgSlot):
                raise TypeError(f"{cls.__name__} missing required value slot {nm!r}")
        for nm, sref in t.sub_slots.items():
            if nm in bindings:
                subs[sref.slot] = bindings[nm]  # the bound sub-graph bundle
            else:
                raise TypeError(f"{cls.__name__} missing required sub-model {nm!r}")
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
        obj._setup(handle_refs, bind(t.output, subs), subgraphs)
        return obj


@dataclass_transform(field_specifiers=(arg, op_slot, sub_slot))
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

    def collect(self, site_type: type) -> list[ValueRef]:
        """Bubble up: nodes in this graph's cone whose producer is ``site_type``."""
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
