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

import torch.nn as nn
from attrs import define, field

import saeco.core as cl
from saeco_research.cgraph2.funcstd import standardize

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
        elif isinstance(p, nn.Module):
            module.add_module(names[n], p)


def _eval_node(n: ValueRef, args: list, cache: cl.Cache, name: str, kwargs: dict):
    """Run one node's producer. ``cl.Module`` producers are threaded a named
    subcache; bare ``nn.Module`` / ``Op`` producers (e.g. the ``nn.Linear``
    encoder, arithmetic ops) are called positionally."""
    p = n.producer
    if isinstance(p, cl.Module):
        return p(*args, cache=cache[name], **kwargs)
    if isinstance(p, nn.Parameter):
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


class ValueGraph(cl.Module):
    """Runnable over a set of target handles (``output`` plus any ``extra``). A
    single ``forward_values`` pass evaluates the **union** of their cones with
    shared memoization -- each interned node (encoder, normalizer, ...) runs at
    most once no matter how many requested outputs depend on it, and only nodes
    actually requested are computed (minimal cone union). ``extra`` are handles
    you want runnable/registered (their producers join the module tree) but that
    aren't the default forward output.

        with vg.forward_values(x, cache=cache) as save:
            recon = save(recon_handle)
            acts  = save(acts_handle)

    ``save`` maps a handle ``ValueRef`` to its realized value ``T`` (the type
    system reads it as ``Ref[T] -> T``); the work is done lazily+memoized inside
    the ``with`` block, so it is one fused graph pass over the cone union."""

    def __init__(self, output: ValueRef, extra: tuple[ValueRef, ...] = ()):
        super().__init__()
        self._output = output
        order = _toposort(output, *extra)
        self._names = _names(order)
        _register_producers(self, order, self._names)

    @property
    def output_ref(self) -> ValueRef:
        return self._output

    def forward_values(self, x, *, cache: cl.Cache, **kwargs) -> _SaveSession:
        return _SaveSession(self, x, cache, kwargs)

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        with self.forward_values(x, cache=cache, **kwargs) as save:
            return save(self.output_ref)


@define
class _SaveSession:
    """Context manager backing ``ValueGraph.forward_values``. Yields a ``save``
    callable; each ``save(ref)`` evaluates that ref's cone, memoizing shared
    upstream nodes across calls so the whole ``with`` block is one fused pass."""

    graph: ValueGraph
    x: object
    cache: cl.Cache
    kwargs: dict
    results: dict = field(init=False)

    @results.default
    def _seed_results(self) -> dict:
        return {Input: self.x}

    def __enter__(self):
        return self.save

    def __exit__(self, *exc):
        return False

    def save(self, target: ValueRef):
        names = self.graph._names
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
