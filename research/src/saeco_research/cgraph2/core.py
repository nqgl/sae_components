"""cgraph v2 — pure relational, interned graphs (task #7).

The v1 `GraphDef` was a mutable builder doing two jobs: accumulating nodes and
interning them (hash-consing). v2 drops the builder. A **node is an immutable
value** ``(producer, inputs)`` that is **interned by structure** in a global
table -- so two structurally-identical sub-DAGs are literally the *same* object,
and fusion is automatic and global, with no per-builder bookkeeping and no
inline/re-emit. There is one global ``Input`` node, so every graph that reads the
input shares it; composing graphs that read the input fuses for free.

  * ``ValueRef``  -- an interned value node. Operators build interned nodes.
  * ``OpRef``     -- an applyable producer handle (module / subgraph / ``Op``).
  * ``Input``     -- the single global input node.
  * ``ref.build()`` / ``Graph`` -- evaluate a target's cone (toposort from it).

Identity == structural equality (because nodes are interned), so consing keys on
producer *identity* (``id`` for modules, the wrapped function's id for ``Op``) --
never a display label.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar

import torch
import torch.nn as nn
from attrs import define
from attrs import field as _attr_field

import saeco.core as cl
from saeco_research.cgraph2.funcstd import standardize

# The forward_values session in scope, so a bubbled-up site (e.g. a freq tracker)
# can defer a stateful update onto it from inside the `with` block.
_ACTIVE_SESSION: ContextVar = ContextVar("cgraph_forward_values", default=None)

# Global intern table: (producer_key, input-node-ids) -> the canonical ValueRef.
_INTERN: dict = {}


class _Input:
    """Sole producer of the global input node."""


INPUT = _Input()


def _callable_name(fn) -> str:
    if isinstance(fn, nn.Module):
        return fn._get_name()
    name = getattr(fn, "__name__", None)
    return name if name and name != "<lambda>" else "op"


@define(eq=False, repr=False)
class Op:
    """A stateless callable producer (arithmetic / edge op). Consed by ``key``
    (a function identity by default), never by the display ``label``."""

    fn: Callable
    label: str
    key: object

    @classmethod
    def create(cls, fn: Callable, label: str | None = None, key=None) -> Op:
        # default consing key is the function's *structure* (standardize), so
        # equivalent inline lambdas fuse without being hoisted to named fns.
        return cls(
            fn, label or _callable_name(fn), key if key is not None else standardize(fn)
        )

    def __call__(self, *args):
        return self.fn(*args)

    def __repr__(self):
        return self.label


def _producer_key(p):
    if p is INPUT:
        return ("input",)
    if isinstance(p, Op):
        return ("op", p.key)
    return ("obj", id(p))  # module / parameter / producer-slot marker


# Arithmetic ops as module-level functions so consing keys on function identity.
def _mul(a, b):
    return a * b


def _add(a, b):
    return a + b


def _sub(a, b):
    return a - b


def _div(a, b):
    return a / b


def _detach(a):
    return a.detach()


@define(frozen=True, eq=False, repr=False)
class ValueRef:
    """An interned value node: ``(producer, inputs)``. Immutable; identity is
    structural (interned). Never construct directly -- use ``ref()`` / operators
    / ``OpRef.__call__`` so it goes through the intern table."""

    producer: object
    inputs: tuple
    name: str | None = None  # display only; not part of identity

    def __mul__(self, o) -> ValueRef:
        return _binop(_mul, "Mul", self, o)

    def __add__(self, o) -> ValueRef:
        return _binop(_add, "Add", self, o)

    def __sub__(self, o) -> ValueRef:
        return _binop(_sub, "Sub", self, o)

    def __truediv__(self, o) -> ValueRef:
        return _binop(_div, "Div", self, o)

    def detach(self) -> ValueRef:
        return ref(Op.create(_detach, "Detach"), (self,))

    def build(self) -> Graph:
        """Freeze a runnable Graph targeting this value (its cone only)."""
        return Graph(self)

    def __repr__(self):
        return f"ValueRef({self.name or _producer_label(self.producer)})"


def ref(producer, inputs: tuple[ValueRef, ...], name: str | None = None) -> ValueRef:
    """Intern a node: return the canonical ValueRef for ``(producer, inputs)``."""
    key = (_producer_key(producer), tuple(id(i) for i in inputs))
    hit = _INTERN.get(key)
    if hit is not None:
        return hit
    v = ValueRef(producer, inputs, name)
    _INTERN[key] = v
    return v


def _binop(fn, label, a: ValueRef, b) -> ValueRef:
    if isinstance(b, ValueRef):
        return ref(Op.create(fn, label), (a, b))
    return ref(Op.create(lambda x: fn(x, b), f"{label}(·, {b})", key=(id(fn), b)), (a,))


# The single global input node (interned once).
Input: ValueRef = ref(INPUT, ())


@define(eq=False, repr=False)
class _Field:
    """Producer that extracts a named field from the (structured) input value.
    The primitive under multi-input graphs (docs/DESIGN-multi-input.md): seed one
    batch struct at ``Input`` and read its fields with ``field("x")`` /
    ``field("target")``. Stateless (nothing to register); fails loud on a missing
    field."""

    name: str

    def __call__(self, batch):
        if isinstance(batch, dict):
            return batch[self.name]  # KeyError if absent -- fail loud
        return getattr(batch, self.name)  # AttributeError if absent -- fail loud

    def __repr__(self):
        return f"field[{self.name}]"


_FIELDS: dict[str, _Field] = {}


def field(name: str) -> ValueRef:
    """Read a named field of the structured input (``field("x")``, ``field(
    "target")``). Seed the input with a dict / struct -- ``forward_values(batch)``
    -- and each field is extracted lazily, so a graph reading only ``field("x")``
    realizes only that, and two reads of the same field fuse to one node.

    The multi-input primitive: it makes a graph able to read *several* distinct
    runtime values (x, target, per-loss weights), which is what a true
    ``total_loss`` handle and transcoders (``y_norm != x_norm``) need. The
    single-tensor ``Input`` path is unchanged -- a graph either reads ``Input`` as
    the value or reads ``field(...)`` off a struct, not both. See
    docs/DESIGN-multi-input.md."""
    f = _FIELDS.get(name)
    if f is None:
        f = _FIELDS[name] = _Field(name)
    return ref(f, (Input,))


@define(eq=False, repr=False)
class OpRef:
    """An applyable producer handle. ``op(x)`` interns a node applying the
    producer to ``x``. A producer slot is an OpRef wrapping a slot marker."""

    producer: object

    def __call__(self, *inputs: ValueRef) -> ValueRef:
        return ref(self.producer, inputs)


def op(producer, *, label: str | None = None) -> OpRef:
    """Wrap a producer (module / subgraph / raw callable) as an applyable OpRef."""
    if not isinstance(producer, nn.Module):
        producer = Op.create(producer, label)
    return OpRef(producer)


def _producer_label(p) -> str:
    if p is INPUT:
        return "input"
    if isinstance(p, _Field):
        return repr(p)
    if isinstance(p, nn.Parameter):
        return "param"
    if isinstance(p, Op):
        return p.label
    if isinstance(p, nn.Module):
        return p._get_name()
    return _callable_name(p)


def _toposort(*targets: ValueRef) -> list[ValueRef]:
    """Reachable-from-targets nodes in dependency order (inputs first)."""
    order: list[ValueRef] = []
    seen: set[int] = set()

    def visit(n: ValueRef):
        if id(n) in seen:
            return
        seen.add(id(n))
        for i in n.inputs:
            visit(i)
        order.append(n)

    for t in targets:
        visit(t)
    return order


def _names(order: list[ValueRef]) -> dict[ValueRef, str]:
    names: dict[ValueRef, str] = {}
    for i, n in enumerate(order):
        names[n] = n.name or ("input" if n.producer is INPUT else f"n{i}")
    return names


def _register_producers(module: cl.Module, order, names) -> None:
    """Register the parameter/module producers in ``order`` on ``module`` so
    ``state_dict`` / ``parameters`` / hook-discovery see them. A producer is
    registered once (interned identity), under its display name."""
    registered: set[int] = set()
    for n in order:
        p = n.producer
        if id(p) in registered:
            continue
        registered.add(id(p))
        if isinstance(p, nn.Parameter):
            module.register_parameter(names[n], p)
        elif isinstance(p, torch.Tensor):
            # a bare buffer used as a value source -- register it so .to()/state_dict
            # see it. (Prefer a Parameter or a buffer-reading module for graph state:
            # a node captures the *tensor*, and .to() reassigns buffers -- see D-04.)
            module.register_buffer(names[n], p)
        elif isinstance(p, nn.Module):
            module.add_module(names[n], p)


def _eval_node(n: ValueRef, args: list, cache: cl.Cache, name: str, kwargs: dict):
    """Run one node's producer. ``cl.Module`` producers are threaded a named
    subcache; bare ``nn.Module`` / ``Op`` producers (e.g. the ``nn.Linear``
    encoder, arithmetic ops) are called positionally."""
    p = n.producer
    if isinstance(p, cl.Module):
        return p(*args, cache=cache[name], **kwargs)
    if isinstance(p, torch.Tensor):  # a parameter or buffer used as a value source
        return p
    return p(*args)


class Graph(cl.Module):
    """Runnable: evaluates a target node's cone. Is-a ``cl.Module`` (state_dict /
    introspection)."""

    def __init__(self, target: ValueRef):
        super().__init__()
        self._target = target
        self._order = _toposort(target)
        self._names = _names(self._order)
        _register_producers(self, self._order, self._names)

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        names = self._names
        results: dict[ValueRef, object] = {Input: x}
        for n in self._order:
            if n is Input:
                continue
            args = [results[i] for i in n.inputs]
            results[n] = _eval_node(n, args, cache, names[n], kwargs)
        return results[self._target]

    def __repr__(self):
        names = self._names
        width = max(len(v) for v in names.values())
        lines = [f"Graph(target={names[self._target]})"]
        for n in self._order:
            ins = ", ".join(names[i] for i in n.inputs)
            arrow = f"  <- {ins}" if ins else ""
            tag = "   [target]" if n is self._target else ""
            label = _producer_label(n.producer)
            lines.append(f"  {names[n]:<{width}} = {label}{arrow}{tag}")
        return "\n".join(lines)


def collect(targets, site_type: type) -> list[ValueRef]:
    """Walk the cone of ``targets`` (apply-style, over the DAG) and return the
    nodes whose producer is an instance of ``site_type``. The graph-native form
    of ``Module.apply`` -- a value 'bubbles up' to the outside without anything
    in between knowing where it lives. The caller derives handles from each
    node's producer (the site) and inputs (e.g. ``site.handles(node.inputs[0])``)."""
    if isinstance(targets, ValueRef):
        targets = (targets,)
    return [n for n in _toposort(*targets) if isinstance(n.producer, site_type)]


@define
class _SaveSession:
    """Context manager backing ``GraphSpec.forward_values``. The yielded object
    is callable -- ``save(ref)`` evaluates that ref's cone, memoizing shared
    upstream nodes across calls so the whole ``with`` block is one fused pass.
    ``save.after(ref, fn)`` defers ``fn(save(ref))`` to block exit -- a bubbled-up
    site (freq tracker) uses it to run its stateful update after the pass.

    ``graph`` is anything with a ``.names`` map (a ``GraphSpec``)."""

    graph: object
    x: object
    cache: cl.Cache
    kwargs: dict
    results: dict = _attr_field(init=False)
    deferred: list = _attr_field(init=False, factory=list)
    token: object = _attr_field(init=False, default=None)

    @results.default
    def _seed_results(self) -> dict:
        return {Input: self.x}

    def __enter__(self):
        self.token = _ACTIVE_SESSION.set(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        try:
            if exc_type is None:
                # materialize every deferred value first (within the still-clean
                # memoized pass), THEN run the side-effecting callbacks -- so one
                # callback can't mutate state a later deferred target reads.
                pending = [(cb, self.save(target)) for target, cb in self.deferred]
                for callback, value in pending:
                    callback(value)
        finally:
            _ACTIVE_SESSION.reset(self.token)
        return False

    def __call__(self, target: ValueRef):
        return self.save(target)

    def after(self, target: ValueRef, callback: Callable) -> None:
        """Defer ``callback(value-of-target)`` until the ``with`` block exits."""
        self.deferred.append((target, callback))

    def save(self, target: ValueRef):
        names = self.graph.names
        results = self.results
        for n in _toposort(target):
            if n in results:
                continue
            args = [results[i] for i in n.inputs]
            name = names.get(n) or f"node{id(n)}"
            results[n] = _eval_node(n, args, self.cache, name, self.kwargs)
        return results[target]


def bind(target: ValueRef, subs: dict) -> ValueRef:
    """Re-intern ``target``'s cone substituting producers/nodes per ``subs`` (a
    map from a slot-marker producer OR a node to its replacement). Because
    re-interning is global, two bindings producing the same structure share
    nodes.

    A producer may define ``__graph_rebind__(self, bound_inputs, subs)`` to
    control how its node re-interns -- the hook a sub-model slot uses to splice
    in a bound sub-graph (see ``spec.sub_slot``)."""
    memo: dict[ValueRef, ValueRef] = {}
    for n in _toposort(target):
        if n in subs:  # a value-slot node (e.g. Input) bound to another ref
            memo[n] = subs[n]
            continue
        rebind = getattr(n.producer, "__graph_rebind__", None)
        if rebind is not None:  # producer-defined re-interning (e.g. sub-model)
            memo[n] = rebind(tuple(memo[i] for i in n.inputs), subs)
            continue
        producer = subs.get(n.producer, n.producer)  # producer-slot substitution
        if producer is INPUT:
            memo[n] = n
            continue
        ins = tuple(memo[i] for i in n.inputs)
        memo[n] = ref(producer, ins, n.name)
    return memo[target]


def at(handle: ValueRef, x: ValueRef) -> ValueRef:
    """Apply a sub-graph to a new input: rebind its ``Input`` to ``x``. The
    graph-native "plug this sub-model in here" move -- e.g. running the SAE on a
    normalized input, or treating a mid-graph node as the input (``decoder_cone``
    does the latter with ``bind(recon, {acts: Input})``)."""
    return bind(handle, {Input: x})
