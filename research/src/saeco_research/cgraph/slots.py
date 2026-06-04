"""Slot-model prototype, explicit API (task #6).

Reuses cgraph.core's executor but presents the slot-model surface:

  * ``ValueRef[T]`` -- a handle to a *value* (a node's output). Has operators.
  * ``OpRef``       -- a handle to a *producer* (module / subgraph / stateless
                       op), APPLYABLE: ``op(*value_refs) -> ValueRef``.

The unification: a value is the nullary case of a producer; a parameter is a
value slot you've pre-bound; an input is a value slot left open. A graph template
is a function ``(g, **producer_slots) -> ValueRef`` -- the producer slots are
ordinary *typed parameters*, so the call signature is type-visible TODAY (no
metaclass), and **binding a slot is ``functools.partial``**. The pretty
``@graph class ...: x: Arg; model: Op`` namespace (via dataclass_transform) is
sugar to layer on top once these semantics are nailed.

Naming: ValueRef/OpRef are the going-forward names (were Ref/Op in core); core's
Ref/Op remain the lower layer until this supersedes them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import torch.nn as nn
from attrs import define, field
from torch import Tensor

from saeco_research.cgraph.core import (
    INPUT,
    Graph,
    Op,
    _assign_names,
    _Node,
    _producer_key,
    _toposort,
)

# Sentinel for "slot has no default" (so None can be a legitimate default).
_NODEFAULT = object()


# Arithmetic / edge ops as module-level functions, so consing keys on FUNCTION
# IDENTITY (id), never on a display label. Two `a * b` fuse because they share
# `_mul`; two ops that merely share a label string never collide.
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


@define(eq=False, repr=False)
class ValueRef[T]:
    """Handle to a value (a node's output). Arithmetic operators build fan-in
    nodes; operands stay shareable, so the result is a DAG, not a tree."""

    _graph: GraphDef
    _node: _Node

    def __mul__(self, o) -> ValueRef[T]:
        return self._graph._binop(_mul, "Mul", self, o)

    def __add__(self, o) -> ValueRef[T]:
        return self._graph._binop(_add, "Add", self, o)

    def __sub__(self, o) -> ValueRef[T]:
        return self._graph._binop(_sub, "Sub", self, o)

    def __truediv__(self, o) -> ValueRef[T]:
        return self._graph._binop(_div, "Div", self, o)

    def detach(self) -> ValueRef[T]:
        return self._graph._unop(_detach, "Detach", self)

    def build(self) -> Graph:
        """Freeze a runnable Graph that targets *this* value -- it computes only
        this ref's cone (toposort from here), so grabbing an intermediate handle
        and building it runs just what's needed to produce it."""
        return self._graph.build(self)

    def __repr__(self):
        return f"ValueRef({self._node.name or '<anon>'})"


@define(eq=False, repr=False)
class OpRef[T]:
    """Applyable handle to a producer (module / subgraph / stateless ``Op``),
    generic over the value type it produces. Pass one as a function parameter to
    leave a producer slot unbound; supply a concrete OpRef to bind it. Applying
    it adds a node and returns a ``ValueRef[T]``.

    If ``_inline`` and the producer is a cgraph ``Graph``, applying it *inlines*
    that graph's nodes into the builder (recursively) instead of adding an opaque
    node -- so two distinct-but-identical frozen subgraphs fuse (stress-test case
    3). Opaque (the default) keeps the subgraph as a black box: a cache-scope
    boundary that prints as a nested block."""

    _graph: GraphDef
    _producer: object
    _inline: bool = False

    def __call__(self, *inputs: ValueRef) -> ValueRef[T]:
        if self._inline and isinstance(self._producer, Graph):
            return self._graph._inline_graph(self._producer, inputs)
        return self._graph._apply(self._producer, inputs)


@define(eq=False)
class GraphDef:
    """Slot-aware builder. ``input()`` / ``param()`` make value slots, ``op()``
    makes producer slots; both hash-cons so independent subgraphs sharing a
    builder fuse."""

    _nodes: list = field(factory=list)
    _cse: dict = field(factory=dict)
    _input_node: _Node = field(factory=lambda: _Node(INPUT, ()))

    def __attrs_post_init__(self):
        self._nodes.append(self._input_node)

    @overload
    def __call__(self, x: Tensor | nn.Parameter, /) -> ValueRef: ...
    @overload
    def __call__[T](
        self,
        producer: Callable[..., T],
        /,
        *,
        label: str | None = ...,
        inline: bool = ...,
    ) -> OpRef[T]: ...
    @overload
    def __call__[T](
        self, producer: Callable[..., T], *inputs: ValueRef, inline: bool = ...
    ) -> ValueRef[T]: ...
    def __call__(self, producer, *inputs, label=None, inline=False):
        """Unified entry point, dispatched on what you pass:
        - ``g(value)`` (a Tensor / Parameter)        -> a ``ValueRef`` (value node)
        - ``g(producer)`` (a module / callable)      -> an ``OpRef`` (apply it later)
        - ``g(producer, *inputs)``                   -> apply now -> ``ValueRef``
        """
        if not inputs:
            if isinstance(producer, (Tensor, nn.Parameter)) and not callable(producer):
                return self._apply(producer, ())  # value node
            return self.op(producer, label=label, inline=inline)  # producer handle
        return self.op(producer, label=label, inline=inline)(*inputs)  # apply

    def input(self) -> ValueRef[Tensor]:
        # One shared value-input slot (the forward arg). Idempotent, so
        # independently-built subgraphs that read g.input() share it and fuse.
        return ValueRef(self, self._input_node)

    def param[T](self, value: T) -> ValueRef[T]:
        """A value slot statically bound to a parameter (a source value node)."""
        return self._apply(value, ())

    def op[T](
        self,
        producer: Callable[..., T],
        *,
        label: str | None = None,
        inline: bool = False,
    ) -> OpRef[T]:
        """Wrap a producer (module / subgraph / raw callable) as an applyable
        OpRef. ``inline=True`` (cgraph ``Graph`` producers only) flattens the
        subgraph into the builder on application so it fuses with siblings."""
        if not isinstance(producer, nn.Module):  # raw callable -> stateless Op
            producer = Op.create(producer, label)
        return OpRef(self, producer, inline)

    def _inline_graph(self, graph: Graph, inputs: tuple[ValueRef, ...]) -> ValueRef:
        """Replay a built Graph's nodes into self (recursively inlining nested
        inlined graphs), hash-consing -> identical frozen subgraphs fuse."""
        memo: dict[int, ValueRef] = {}
        graph_input = next(n for n in graph._order if n.producer is INPUT)
        memo[id(graph_input)] = inputs[0]
        for node in graph._order:
            if node.producer is INPUT:
                continue
            ins = tuple(memo[id(i)] for i in node.inputs)
            if isinstance(node.producer, Graph):
                memo[id(node)] = self._inline_graph(node.producer, ins)
            else:
                memo[id(node)] = self._apply(node.producer, ins)
        return memo[id(graph._output)]

    def _apply(self, producer, inputs: tuple[ValueRef, ...]) -> ValueRef:
        # A cgraph Graph is OUR transparent structure -- inline it (recursively)
        # so its shared nodes hash-cons with the rest of the builder, rather than
        # adding an opaque node. Two distinct frozen subgraphs that share an
        # encoder thus fuse (the `Nest` case).
        if isinstance(producer, Graph):
            return self._inline_graph(producer, inputs)
        in_nodes = tuple(r._node for r in inputs)
        key = (_producer_key(producer), tuple(id(n) for n in in_nodes))
        hit = self._cse.get(key)
        if hit is not None:
            return ValueRef(self, hit)
        node = _Node(producer, in_nodes)
        self._nodes.append(node)
        self._cse[key] = node
        return ValueRef(self, node)

    def _unop(self, fn: Callable, label: str, a: ValueRef) -> ValueRef:
        return self._apply(Op.create(fn, label), (a,))  # key defaults to id(fn)

    def _binop(self, fn: Callable, label: str, a: ValueRef, b) -> ValueRef:
        if isinstance(b, ValueRef):
            return self._apply(Op.create(fn, label), (a, b))
        # scalar / tensor constant on the rhs -> fold into a unary op, keyed on
        # (function identity, constant) -- still never a display string
        return self._apply(
            Op.create(lambda x: fn(x, b), f"{label}(·, {b})", key=(id(fn), b)), (a,)
        )

    def build(self, output: ValueRef) -> Graph:
        order = _toposort(output._node)
        return Graph(order=order, names=_assign_names(order), output=output._node)

    # --- template support (used by the @graph namespace in namespace.py) ---
    # A template is a GraphDef recording nodes whose producers/inputs may be
    # *slot markers*; `inline` replays it into another builder with the slots
    # bound, hash-consing as it goes (so nested templates fuse).

    def arg_slot(self, default=_NODEFAULT) -> ValueRef:
        """A value-input slot, optionally with a default value used when the slot
        is left unbound at construction."""
        node = _Node(_ArgSlot(default), ())
        self._nodes.append(node)
        return ValueRef(self, node)

    def op_slot(self, default=_NODEFAULT) -> OpRef:
        """A producer slot, optionally with a default producer used when the slot
        is left unbound at construction."""
        return OpRef(self, _OpSlot(default))

    def inline(
        self,
        target: GraphDef,
        *,
        values: dict[_Node, ValueRef],
        producers: dict[object, object],
    ) -> dict[_Node, ValueRef]:
        """Replay ALL of this template's nodes into ``target``, binding slots, and
        return the node->ValueRef map (so the caller can expose every named
        handle, not just the output). ``values`` maps a value-slot node to its
        bound/defaulted ValueRef; ``producers`` maps a producer-slot *identity* (a
        marker, or a concrete default producer) to its substitute. Re-emitting via
        ``target._apply`` hash-conses, so nested templates fuse. Inlining the whole
        template is cheap (structure only) -- nothing runs until a handle is built,
        which toposorts from that handle and so computes only its cone."""
        memo: dict[_Node, ValueRef] = {}
        for node in self._nodes:
            if node.producer is INPUT:
                continue  # template's unused bare input slot
            if node in values:
                memo[node] = values[node]
                continue
            ins = tuple(memo[i] for i in node.inputs)
            prod = producers.get(node.producer, node.producer)
            memo[node] = target._apply(prod, ins)
        return memo


class _ArgSlot:
    """Marker producer for a value-input slot of a template; ``default`` (if not
    _NODEFAULT) is the value used when the slot is left unbound."""

    def __init__(self, default=_NODEFAULT):
        self.default = default


class _OpSlot:
    """Marker producer for a producer slot of a template; ``default`` (if not
    _NODEFAULT) is the producer used when the slot is left unbound."""

    def __init__(self, default=_NODEFAULT):
        self.default = default
