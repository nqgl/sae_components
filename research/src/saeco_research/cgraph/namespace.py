"""dataclass_transform class-namespace sugar over the explicit slot layer (#6).

Lets you write a graph template as a class whose *slots are annotated fields*:

    class Loss(GraphSpec):
        x: ValueRef = arg()        # value-input slot
        model: OpRef = op_slot()   # producer slot
        compare: OpRef = op_slot() # producer slot
        y_pred = model(x)          # wire the body
        output = compare(x, y_pred)

`@dataclass_transform` makes construction type-visible: pyright sees
`Loss(x: ValueRef, model: OpRef, compare: OpRef)` and checks/rejects bindings.
Constructing **binds the slots and inlines the template into the builder of the
bound refs** (so nested templates fuse by hash-consing -- see fusion_stress.py).
The instance exposes `.out` and composes with `+`/`*` (delegating to it).

This is sugar over slots.py: `arg()`/`op_slot()` create the same slot handles the
explicit `(g, **slots)` form uses; the body wiring is identical. The only "magic"
is the metaclass `__prepare__` that opens a recording builder before the body.
(Annotation-only slots -- `x: ValueRef` with no `= arg()` -- would need a
`__missing__` namespace; deferred. The `= arg()` form needs no such magic.)
"""

from __future__ import annotations

from typing import Any, dataclass_transform

from saeco_research.cgraph.slots import (
    _NODEFAULT,
    Graph,
    GraphDef,
    OpRef,
    ValueRef,
    _ArgSlot,
    _OpSlot,
)

# Annotation aliases matching the slot vocabulary used in sketches.
Arg = ValueRef  # a value-input slot
Op = OpRef  # a producer slot

# Recording builder open during a @graph class body (set by the metaclass).
_CURRENT: GraphDef | None = None

# Module-level placeholder so a bare `g(...)` in a @graph body type-checks
# (pyright resolves the name to this). At runtime the metaclass injects the real
# per-class recording builder into the body namespace, shadowing this -- so
# `compare: OpRef = g(nn.ReLU())` reads naturally and is type-visible.
g: GraphDef = GraphDef()


def arg(default=_NODEFAULT) -> Any:
    """Field specifier + runtime handle for a value-input slot. Passing
    ``default=`` makes the slot optional (the field specifier's ``default`` is
    what dataclass_transform reads to mark it optional)."""
    assert _CURRENT is not None, "arg() is only valid inside a GraphSpec class body"
    return _CURRENT.arg_slot(default)


def op_slot(default=_NODEFAULT) -> Any:
    """Field specifier + runtime handle for a producer slot. ``default=`` (a
    producer) makes it optional, e.g. ``compare = op_slot(default=nn.ReLU())``."""
    assert _CURRENT is not None, "op_slot() is only valid inside a GraphSpec class body"
    return _CURRENT.op_slot(default)


class _Template:
    def __init__(self, output, value_slots, op_slots, handles):
        self.output: ValueRef = output
        self.value_slots: dict[str, ValueRef] = value_slots  # name -> slot's ValueRef
        self.op_slots: dict[str, OpRef] = op_slots  # name -> slot's OpRef
        self.handles: dict[str, ValueRef] = handles  # every named value (grabbable)


@dataclass_transform(field_specifiers=(arg, op_slot))
class _GraphMeta(type):
    @classmethod
    def __prepare__(cls, name, bases, **kw):
        global _CURRENT
        _CURRENT = GraphDef()
        return {"g": _CURRENT}  # inject the recording builder as `g`

    def __new__(mcs, name, bases, ns, **kw):
        global _CURRENT
        _CURRENT = None  # close the recording builder
        # drop the injected builder and all template refs (slots/handles) from the
        # class dict, so instances serve inlined handles via __getattr__ instead of
        # resolving to the template-time class attribute.
        body = {
            k: v
            for k, v in ns.items()
            if k != "g" and not isinstance(v, (ValueRef, OpRef))
        }
        cls = super().__new__(mcs, name, bases, body)
        if not any(isinstance(b, _GraphMeta) for b in bases):
            return cls  # the GraphSpec base itself -- nothing to capture
        # A slot is any *annotated* field holding a ValueRef (value slot) or OpRef
        # (producer slot) -- whether it came from arg()/op_slot() (a marker) or a
        # concrete `g(...)` default. The annotation is what marks it a slot, so the
        # name<->ref link is just `vars(cls)`.
        value_slots, op_slots = {}, {}
        for slot_name in ns.get("__annotations__", {}):
            if slot_name == "output" or slot_name.startswith("_"):
                continue
            v = ns.get(slot_name)
            if isinstance(v, OpRef):
                op_slots[slot_name] = v
            elif isinstance(v, ValueRef):
                value_slots[slot_name] = v
        handles = {k: v for k, v in ns.items() if isinstance(v, ValueRef)}
        output = ns.get("output")
        if not isinstance(output, ValueRef):
            raise TypeError(f"{name} must designate `output = <ValueRef>` in its body")
        cls._template = _Template(output, value_slots, op_slots, handles)
        return cls

    def __call__(cls, **bindings):
        t: _Template = cls._template
        known = set(t.value_slots) | set(t.op_slots)
        for name in bindings:
            if name not in known:
                raise TypeError(f"{cls.__name__} has no slot {name!r}")
        target: GraphDef | None = next(
            (v._graph for v in bindings.values() if hasattr(v, "_graph")), None
        )
        if target is None:
            raise TypeError(f"{cls.__name__}(...) needs at least one bound slot")

        # value slots: bound -> override; arg()-marker with default -> default value;
        # concrete `g(value)` default -> left for inline to re-emit.
        values: dict[object, ValueRef] = {}
        for name, vref in t.value_slots.items():
            node = vref._node
            slot = node.producer
            if name in bindings:
                b = bindings[name]
                values[node] = b if isinstance(b, ValueRef) else target.param(b)
            elif isinstance(slot, _ArgSlot):
                if slot.default is _NODEFAULT:
                    raise TypeError(f"{cls.__name__} missing required slot {name!r}")
                values[node] = target.param(slot.default)

        # producer slots: keyed by the slot's *producer identity* (marker OR the
        # concrete default producer). Bound -> substitute; op_slot()-marker with
        # default -> substitute the default; concrete default -> left as-is.
        producers: dict[object, object] = {}
        for name, oref in t.op_slots.items():
            identity = oref._producer
            if name in bindings:
                b = bindings[name]
                ob = b if isinstance(b, OpRef) else target.op(b)
                producers[identity] = ob._producer
            elif isinstance(identity, _OpSlot):
                if identity.default is _NODEFAULT:
                    raise TypeError(f"{cls.__name__} missing required slot {name!r}")
                producers[identity] = target.op(identity.default)._producer

        memo = t.output._graph.inline(target, values=values, producers=producers)
        obj = object.__new__(cls)
        # expose every named value of the template as a grabbable handle, mapped
        # into the target builder (so they fuse + can each be built/spliced)
        obj._handles = {
            name: memo[ref._node]
            for name, ref in t.handles.items()
            if ref._node in memo
        }
        obj._out = memo[t.output._node]
        return obj


@dataclass_transform(field_specifiers=(arg, op_slot))
class GraphSpec(metaclass=_GraphMeta):
    """Base for `@graph`-style templates. A subclass declares slots as annotated
    fields (`= arg()` / `= op_slot()`), wires the body, and sets `output`.
    Constructing it binds the slots and emits into the builder of the bound refs;
    the instance is a value (`.out`) that composes with `+`/`*`."""

    # declared via __slots__ (not annotations) so dataclass_transform does NOT
    # treat them as construction fields -- only the subclass's slots are fields.
    __slots__ = ("_out", "_handles")

    @property
    def out(self) -> ValueRef:
        return self._out

    def __getattr__(self, name) -> ValueRef:
        # named values from the body (e.g. `acts`, `recon`) are grabbable handles
        try:
            handles = object.__getattribute__(self, "_handles")
        except AttributeError:
            raise AttributeError(name) from None
        if name in handles:
            return handles[name]
        raise AttributeError(name)

    def produce(self, ref: ValueRef) -> Graph:
        """Freeze a runnable Graph targeting one of this spec's handles -- it runs
        only that handle's cone (e.g. ``sae.produce(sae.acts)`` skips the decoder)."""
        return ref.build()

    def __add__(self, o) -> ValueRef:
        return self._out + (o.out if isinstance(o, GraphSpec) else o)

    def __mul__(self, o) -> ValueRef:
        return self._out * (o.out if isinstance(o, GraphSpec) else o)

    def __sub__(self, o) -> ValueRef:
        return self._out - (o.out if isinstance(o, GraphSpec) else o)

    def __truediv__(self, o) -> ValueRef:
        return self._out / (o.out if isinstance(o, GraphSpec) else o)
