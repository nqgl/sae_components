"""Prototype: a DAG composition primitive for `saeco.core` components.

Exploratory scratch (research/). This generalizes the linear `Propagator`
(Seq / Parallel / Router) to an arbitrary directed acyclic graph of named
nodes. Each node names which other nodes' outputs feed it; the executor
evaluates in topological order and stores results keyed by node identity, so a
node referenced by two consumers runs *exactly once*.

That last property is the whole point: it is the job `cl.ReuseForward` does
today via an ``(id(module), id(x))`` cache key. In a tree (Seq/Parallel) a
shared subcomputation has to be memoized; in a DAG it is just a node with two
out-edges, computed once because the executor dedupes by node identity. So the
graph does not *add* a feature -- it removes the reason ReuseForward exists.

On top of that, the builder hash-conses: ``g(producer, *inputs)`` returns an
existing equivalent node, so independently-authored subgraphs sharing a builder
fuse automatically (the encoder shared by a main + aux model becomes one node).

Surface
-------
    g = GraphDef()
    @g.namespace
    class model:
        x   = g.input
        h   = g(encoder, x)          # computed once even if used twice
        g.output = g(decoder, h * gate)   # operators build arithmetic fan-in nodes
    # `model` is now a Graph (a cl.Module). `g.build(ref)` is the non-namespace path.

Naming is optional sugar (execution runs on node identity, not names): the
``@g.namespace`` decorator learns each node's name from the variable it was bound
to, so refs stay plain value objects.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from attrs import define, field

import saeco.core as cl

# Sentinel producer marking the graph's input node (value supplied at forward).
INPUT = object()


@define(eq=False)
class _Node:
    """Internal graph node. ``producer`` is INPUT | nn.Module | nn.Parameter |
    plain callable; ``inputs`` are the parent nodes whose outputs feed it.

    ``eq=False`` keeps object identity for hashing/equality -- nodes are keyed by
    identity in the executor's results map and in the consing table."""

    producer: object
    inputs: tuple = ()
    name: str | None = None


def _callable_name(fn) -> str:
    """Best-effort display name for a producer callable."""
    if isinstance(fn, nn.Module):
        return fn._get_name()
    name = getattr(fn, "__name__", None)
    return name if name and name != "<lambda>" else "op"


@define(eq=False, repr=False)
class Op:
    """A small named, stateless operation for arithmetic / edge nodes.

    Kept deliberately separate from ``cl.Module``: Ops have no parameters and no
    cache, so the executor calls them positionally (``op(*args)``). Build with
    :meth:`create` so the label / consing-key derive themselves."""

    fn: Callable
    label: str
    key: object  # stable identity for hash-consing

    @classmethod
    def create(cls, fn: Callable, label: str | None = None, key=None) -> Op:
        # label -> printout name; key -> consing identity. Built-in arithmetic
        # passes its kind ("Mul", ...) as key so two `a*b` ops fuse; a wrapped
        # callable falls back to the function object's id (so two `Op.create(
        # torch.exp)` share a key).
        return cls(
            fn=fn,
            label=label or _callable_name(fn),
            key=key if key is not None else id(fn),
        )

    def __call__(self, *args):
        return self.fn(*args)

    def __repr__(self):
        return self.label


@define(eq=False, repr=False)
class Ref[T]:
    """A handle to a node's output, generic over that output's type.

    Arithmetic operators build fan-in nodes (``a * b`` -> a ``Mul`` node reading
    both ``a`` and ``b``); the operands stay shareable, so the resulting graph
    is a DAG, not a tree. ``.split(n)`` and ``.detach()`` are per-edge transforms
    expressed as their own nodes."""

    _graph: GraphDef
    _node: _Node

    def __mul__(self, other) -> Ref:
        return self._graph._binop("Mul", lambda a, b: a * b, self, other)

    def __add__(self, other) -> Ref:
        return self._graph._binop("Add", lambda a, b: a + b, self, other)

    def __sub__(self, other) -> Ref:
        return self._graph._binop("Sub", lambda a, b: a - b, self, other)

    def __truediv__(self, other) -> Ref:
        return self._graph._binop("Div", lambda a, b: a / b, self, other)

    def detach(self) -> Ref[T]:
        return self._graph._unop("Detach", lambda a: a.detach(), self)

    def split(self, n: int) -> tuple[Ref, ...]:
        """Turn a tuple-valued ref into ``n`` single-output refs (getitem nodes)."""
        return tuple(
            self._graph._unop(f"GetItem[{i}]", _getter(i), self) for i in range(n)
        )

    def __repr__(self):
        return f"Ref({self._node.name or '<anon>'})"


def _getter(i: int) -> Callable:
    return lambda a: a[i]


def _producer_key(p):
    """Canonical identity for hash-consing. Modules / parameters / raw callables
    key on instance identity; Ops key on their (stable) op kind, so two `a + b`
    nodes with the same inputs fuse but distinct modules never do."""
    if isinstance(p, Op):
        return ("op", p.key)
    return ("obj", id(p))


@define(eq=False)
class GraphDef:
    """Mutable builder. Accumulates nodes (deduping equivalent ones) as you
    reference them; finalize via ``@g.namespace`` (designating ``g.output``) or
    ``g.build(ref)``."""

    _input: _Node = field(factory=lambda: _Node(INPUT, ()))
    _nodes: list = field(factory=list)
    _cse: dict = field(factory=dict)  # (producer_key, input_node_ids) -> node
    output: Ref | None = None  # designated inside an @g.namespace body

    def __attrs_post_init__(self):
        self._nodes.append(self._input)

    @property
    def input(self) -> Ref:
        return Ref(self, self._input)

    def __call__(self, producer, *inputs: Ref, fresh: bool = False) -> Ref:
        """Add a node, or — by default — return an existing equivalent one
        (hash-consing). Two calls with the same producer and the same input
        nodes collapse to one node, so independently-authored subgraphs sharing
        a builder fuse automatically. Pass ``fresh=True`` for an impure producer
        (stochastic / stateful) that must run on its own each time."""
        in_nodes = tuple(r._node for r in inputs)
        key = None
        if not fresh:
            key = (_producer_key(producer), tuple(id(n) for n in in_nodes))
            hit = self._cse.get(key)
            if hit is not None:
                return Ref(self, hit)
        node = _Node(producer, in_nodes)
        self._nodes.append(node)
        if key is not None:
            self._cse[key] = node
        return Ref(self, node)

    def _unop(self, label: str, fn: Callable, a: Ref) -> Ref:
        return self(Op.create(fn, label, key=label), a)

    def _binop(self, label: str, fn: Callable, a: Ref, b) -> Ref:
        if isinstance(b, Ref):
            return self(Op.create(fn, label, key=label), a, b)
        # scalar / tensor constant on the rhs -> fold into a unary op (the
        # constant is baked into the key so identical folds still fuse)
        return self._unop(f"{label}(·, {b})", lambda x: fn(x, b), a)

    def namespace(self, cls) -> Graph:
        """Class-as-namespace decorator. Learns each node's name from the
        variable it was bound to (definition order is preserved, so naming is
        deterministic), then builds and *returns the Graph* using the
        ``g.output`` designated in the body. Fails loudly if no output was set."""
        for name, val in vars(cls).items():
            if isinstance(val, Ref):
                val._node.name = name
        if self.output is None:
            raise ValueError(
                "designate the graph output with `g.output = <ref>` inside the "
                "@g.namespace body"
            )
        return self.build(self.output)

    def build(self, output: Ref) -> Graph:
        order = _toposort(output._node)
        return Graph(order=order, names=_assign_names(order), output=output._node)


def _toposort(output: _Node) -> list[_Node]:
    """Reachable-from-output nodes in topological order (inputs first). Raises on
    a cycle -- fail loud at the boundary."""
    order: list[_Node] = []
    temp: set[int] = set()
    perm: set[int] = set()

    def visit(n: _Node):
        if id(n) in perm:
            return
        if id(n) in temp:
            raise ValueError("cycle detected in graph")
        temp.add(id(n))
        for inp in n.inputs:
            visit(inp)
        temp.discard(id(n))
        perm.add(id(n))
        order.append(n)

    visit(output)
    return order


def _assign_names(order: list[_Node]) -> dict[_Node, str]:
    """Resolve names into a map: keep any set by @namespace, auto-index the rest.
    Names live here, not on the nodes -- nodes stay immutable and reusable."""
    names: dict[_Node, str] = {}
    seen: dict[str, _Node] = {}
    for i, node in enumerate(order):
        name = node.name or ("input" if node.producer is INPUT else f"n{i}")
        if seen.setdefault(name, node) is not node:
            raise ValueError(f"duplicate node name {name!r} in graph")
        names[node] = name
    return names


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


class Graph(cl.Module):
    """A frozen, runnable DAG. Is-a ``cl.Module``, so it nests as a node in a
    larger graph and participates in ``state_dict`` / module introspection.
    Build via ``GraphDef.build`` / ``@g.namespace`` rather than directly."""

    def __init__(self, *, order: list[_Node], names: dict[_Node, str], output: _Node):
        super().__init__()
        self._order = order
        self._names = names
        self._output = output
        # Register parametric children so state_dict / .parameters() work.
        for node in order:
            p = node.producer
            if isinstance(p, nn.Parameter):
                self.register_parameter(names[node], p)
            elif isinstance(p, nn.Module):
                self.add_module(names[node], p)

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        names = self._names
        results: dict[_Node, object] = {}
        for node in self._order:
            p = node.producer
            if p is INPUT:
                results[node] = x
                continue
            args = [results[inp] for inp in node.inputs]
            if isinstance(p, cl.Module):
                out = p(*args, cache=cache[names[node]], **kwargs)
            elif isinstance(p, nn.Parameter):
                out = p
            else:  # Op / plain callable / non-cl nn.Module (e.g. nn.ReLU)
                out = p(*args)
            cache._write(names[node], out)
            results[node] = out
        return results[self._output]

    def __repr__(self):
        return self.render()

    def render(self, *, recurse: bool = True, _indent: int = 0) -> str:
        """Adjacency listing. By default recurses into subgraph (Graph-producer)
        nodes, printing each in an indented block; pass ``recurse=False`` for a
        flat one-level view."""
        names = self._names
        pad = "  " * _indent
        width = max(len(n) for n in names.values())
        lines = [f"{pad}Graph(output={names[self._output]})"]
        for n in self._order:
            ins = ", ".join(names[i] for i in n.inputs)
            arrow = f"  <- {ins}" if ins else ""
            tag = "   [output]" if n is self._output else ""
            label = _producer_label(n.producer)
            lines.append(f"{pad}  {names[n]:<{width}} = {label}{arrow}{tag}")
            if recurse and isinstance(n.producer, Graph):
                lines.append(n.producer.render(recurse=True, _indent=_indent + 2))
        return "\n".join(lines)
