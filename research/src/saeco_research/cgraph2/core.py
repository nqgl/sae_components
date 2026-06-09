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

import functools
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, cast, overload

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
_INTERN: dict[tuple[tuple, tuple[int, ...]], ValueRef] = {}

# A substitution map for `bind`. Deliberately heterogeneous: a key is EITHER a
# slot-marker / producer object (substitute the producer) OR a `ValueRef` node
# (substitute the whole node, e.g. `Input -> x`); the value is the matching
# replacement producer or node. The two roles share one map so a single `bind`
# pass resolves both -- hence `object`, not a tidy homogeneous type.
type Subs = dict[object, object]


class _Input:
    """Sole producer of the global input node."""


INPUT = _Input()


def _callable_name(fn) -> str:
    if isinstance(fn, nn.Module):
        return fn._get_name()
    name = getattr(fn, "__name__", None)
    return (
        name
        if name and name != "<lambda>"
        else (
            str(clsname)
            if (clsname := getattr(type(fn), "__name__", None))
            else str(type(fn))
        )
    )


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
class ValueRef[V = Any, P = Any]:
    """An interned value node: ``(producer, inputs)``. Immutable; identity is
    structural (interned). Two type params, both defaulting to ``Any``: ``V`` is the
    *value* the node yields at runtime (``ValueRef[Tensor]`` -- carried by
    ``ModuleRef.get(lambda n: n.scale)``), and ``P`` is the *producer* type (so
    ``collect(FreqTracker)`` yields ``ValueRef[Any, FreqTracker]`` and ``site.producer``
    is a ``FreqTracker``). Never construct directly -- use ``ref()`` / operators /
    ``OpRef.__call__`` so it goes through the intern table."""

    producer: P
    inputs: tuple
    name: str | None = None  # display only; not part of identity

    def __mul__(self, o) -> ValueRef[torch.Tensor]:
        return _binop(_mul, "Mul", self, o)

    def __add__(self, o) -> ValueRef[torch.Tensor]:
        return _binop(_add, "Add", self, o)

    def __sub__(self, o) -> ValueRef[torch.Tensor]:
        return _binop(_sub, "Sub", self, o)

    def __truediv__(self, o) -> ValueRef[torch.Tensor]:
        return _binop(_div, "Div", self, o)

    def detach(self) -> ValueRef[torch.Tensor]:
        return ref(Op.create(_detach, "Detach"), (self,))

    def __call__(self, *args):
        hook = getattr(self.producer, "__partial_call__", None)
        if hook is None:
            raise TypeError(f"{self!r} is not callable")
        return hook(self, *args)

    def build(self) -> Graph:
        """Freeze a runnable Graph targeting this value (its cone only)."""
        return Graph(self)

    def __repr__(self):
        return f"ValueRef({self.name or _producer_label(self.producer)})"


def ref_as[P](
    node_cls: type,
    producer: P,
    inputs: tuple[ValueRef, ...],
    name: str | None = None,
) -> ValueRef[Any, P]:
    """Intern a node, constructing it as ``node_cls`` (a ``ValueRef`` subclass) on a
    miss. The interning key is structural (producer + input identities), independent
    of the class -- so ``node_cls`` only decides what an *uninterned* node becomes.
    Used for the ``_AppliedSlot`` forward node, which is a ``ValueRef`` carrying a
    ``__getattr__`` for sub-graph handle access (``m(x).recon``)."""
    key = (_producer_key(producer), tuple(id(i) for i in inputs))
    hit = _INTERN.get(key)
    if hit is not None:
        return hit
    v = node_cls(producer, inputs, name)
    _INTERN[key] = v
    return v


def ref[P](
    producer: P, inputs: tuple[ValueRef, ...], name: str | None = None
) -> ValueRef[Any, P]:
    """Intern a node: return the canonical ValueRef for ``(producer, inputs)``."""
    return ref_as(ValueRef, producer, inputs, name)


def _binop(fn, label, a: ValueRef, b) -> ValueRef[torch.Tensor]:
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


@overload
def field(spec: str) -> ValueRef[Any]: ...
@overload
def field[T](spec: T) -> ValueRef[T]: ...
def field(spec: Any) -> ValueRef:
    """Read a named field of the structured input. Pass a field name (``field(
    "target")``) or a typed ``DictBatch`` field (``field(Batch.target)``, which
    reads as the field's value type to the checker -- see ``cgraph2.dictbatch``).

    A typed ``field(Batch.target)`` carries the field's value type through:
    ``Batch.target`` reads as ``torch.Tensor`` (the annotation), so ``field(
    Batch.target)`` is a ``ValueRef[torch.Tensor]``; a bare ``field("name")`` string
    stays ``ValueRef[Any]``.

    Seed the input with a dict / struct -- ``forward_values(batch)`` -- and each
    field is extracted lazily, so a graph reading only ``field("x")`` realizes only
    that, and two reads of the same field fuse to one node.

    The multi-input primitive: it makes a graph able to read *several* distinct
    runtime values (x, target, per-loss weights), which is what a true
    ``total_loss`` handle and transcoders (``y_norm != x_norm``) need. The
    single-tensor ``Input`` path is unchanged -- a graph either reads ``Input`` as
    the value or reads ``field(...)`` off a struct, not both. See
    docs/DESIGN-multi-input.md."""
    name = spec if isinstance(spec, str) else spec.name  # str | FieldRef
    f = _FIELDS.get(name)
    if f is None:
        f = _FIELDS[name] = _Field(name)
    return ref(f, (Input,))


@define(eq=False, repr=False)
class OpRef:
    """An applyable producer handle. ``op(x)`` interns a node applying the
    producer to ``x``. A producer slot is an OpRef wrapping a slot marker."""

    producer: object

    def __call__(self, *inputs: ValueRef) -> ValueRef[torch.Tensor]:
        # graph ops yield tensors (this is a tensor-computation graph), so the value
        # type is Tensor by default -- a non-tensor op would read as Tensor, but none
        # exist here. (`.get(fn)` carries a real R for attribute reads.)
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


class _TensorSource(nn.Module):
    """A 0-ary producer holding a bound tensor as its *own registered buffer* and
    reading it live at eval -- the device-safe form of a bare-tensor value source.

    The D-04/D-16 model: a node captures a *tensor object*. ``nn.Module._apply``
    (``.to(device)`` / dtype moves) swaps a ``Parameter``'s ``.data`` **in place**
    (identity preserved -- safe to capture) but **reassigns** a registered buffer to
    a new tensor -- so a node that captured the buffer keeps reading the orphaned
    original while the registry moves, a silent wrong-device/dtype. Holding the
    tensor as this module's own buffer and reading ``self.value`` at call time means
    ``.to()`` on the registered graph moves exactly what the node reads."""

    value: torch.Tensor

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer("value", value)

    def forward(self) -> torch.Tensor:
        return self.value

    def _get_name(self) -> str:
        return "tensor"


# one holder per bound tensor (by identity), so two bindings of the same tensor
# fuse to one node -- same shape as the `_FIELDS` cache.
_TENSOR_SOURCES: dict[int, _TensorSource] = {}


def tensor_source(value: torch.Tensor) -> ValueRef[torch.Tensor]:
    """A device-safe 0-ary source node for a bare tensor (see ``_TensorSource``).
    The slot machinery routes every bound/default tensor through this; ``Parameter``
    sources stay raw (in-place-safe and registered as parameters)."""
    holder = _TENSOR_SOURCES.get(id(value))
    if holder is None:
        holder = _TENSOR_SOURCES[id(value)] = _TensorSource(value)
    return ref(holder, ())


def _register_producers(module: cl.Module, order, names) -> None:
    """Register the parameter/module producers in ``order`` on ``module`` so
    ``state_dict`` / ``parameters`` / hook-discovery see them. A producer is
    registered once (interned identity), under its display name.

    A bare non-``Parameter`` tensor producer is refused loudly: registering it as a
    buffer is exactly the D-04 trap (``.to()`` reassigns the registry entry while the
    node keeps reading the captured orphan -- the state_dict copy *lies*). The slot
    machinery never produces one (bound tensors are wrapped in ``_TensorSource``);
    this guards the raw ``ref(tensor, ())`` escape hatch."""
    registered: set[int] = set()
    for n in order:
        p = n.producer
        if id(p) in registered:
            continue
        registered.add(id(p))
        if isinstance(p, nn.Parameter):
            module.register_parameter(names[n], p)
        elif isinstance(p, torch.Tensor):
            raise TypeError(
                f"node {names[n]!r} captures a bare tensor as its producer; "
                "registering it as a buffer would go stale on .to() (the registry "
                "entry is reassigned while the node reads the captured orphan -- "
                "D-04). Bind it through a slot (auto-wrapped device-safe), or use "
                "nn.Parameter / a module reading its own buffer / slot.get(...)."
            )
        elif isinstance(p, nn.Module):
            module.add_module(names[n], p)


def _eval_node(
    n: ValueRef, args: list, cache: cl.Cache, name: str, kwargs: dict[str, Any]
):
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


def collect[S](targets, site_type: type[S]) -> list[ValueRef[Any, S]]:
    """Walk the cone of ``targets`` (apply-style, over the DAG) and return the
    nodes whose producer is an instance of ``site_type`` -- typed ``ValueRef[Any, S]``,
    so ``node.producer`` is an ``S``. The graph-native form of ``Module.apply`` --
    a value 'bubbles up' to the outside without anything in between knowing where
    it lives. The caller derives handles from each node's producer (the site) and
    inputs (e.g. ``site.handles(node.inputs[0])``)."""
    if isinstance(targets, ValueRef):
        targets = (targets,)
    return [n for n in _toposort(*targets) if isinstance(n.producer, site_type)]


# one deferred side-effect: (materialize(session) -> value, run(value), training_only)
type _Deferral = tuple[Callable[["_SaveSession"], Any], Callable[[Any], None], bool]


@define
class _SaveSession:
    """Context manager backing ``GraphSpec.forward_values``. The yielded object
    is callable -- ``save(ref)`` evaluates that ref's cone, memoizing shared
    upstream nodes across calls so the whole ``with`` block is one fused pass.
    ``save.after(ref, fn)`` defers ``fn(save(ref))`` to block exit; a method marked
    :func:`deferred_execution` defers itself the same way. ``training`` gates the
    training-only deferrals (an EMA update runs after a training pass, not eval).

    ``graph`` is anything with a ``.names`` map (a ``GraphSpec``)."""

    graph: object
    x: object
    cache: cl.Cache
    kwargs: dict[str, Any]
    training: bool = True
    results: dict[ValueRef, object] = _attr_field(init=False)
    deferred: list[_Deferral] = _attr_field(init=False, factory=list)
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
                # training-only deferrals (e.g. EMA updates) run only on a training
                # pass, so eval passes don't advance tracker state. Materialize every
                # deferred value first (within the still-clean memoized pass), THEN
                # run the side effects -- so one can't mutate state a later deferred
                # target reads.
                pending = [
                    (run, materialize(self))
                    for materialize, run, training_only in self.deferred
                    if self.training or not training_only
                ]
                for run, value in pending:
                    run(value)
        finally:
            _ACTIVE_SESSION.reset(self.token)
        return False

    def __call__(self, target: ValueRef):
        return self.save(target)

    def after(self, target: ValueRef, callback: Callable) -> None:
        """Defer ``callback(value-of-target)`` until the ``with`` block exits (always,
        on clean exit). For a *training-only* stateful update that auto-defers, mark
        the method :func:`deferred_execution` instead."""
        self.deferred.append((lambda s: s.save(target), callback, False))

    def defer_call(self, method, instance, args, kwargs) -> None:
        """Defer a training-only method call to block exit; ``ValueRef`` arguments
        are materialized in the (memoized) pass. Backs :func:`deferred_execution`."""

        def materialize(s):
            mats = tuple(s.save(a) if isinstance(a, ValueRef) else a for a in args)
            matk = {
                k: (s.save(v) if isinstance(v, ValueRef) else v)
                for k, v in kwargs.items()
            }
            return mats, matk

        def run(materialized):
            mats, matk = materialized
            method(instance, *mats, **matk)

        self.deferred.append((materialize, run, True))

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


def deferred_execution(method):
    """Mark a stateful side-effecting method as a *training callback*: inside a
    ``forward_values`` block it is collected and run at block exit -- only on a
    training pass (``session.training``), with any ``ValueRef`` arguments
    materialized in the memoized pass; outside a session it runs immediately.

    The clean replacement for reaching into ``_ACTIVE_SESSION`` by hand: a freq
    tracker's EMA writes back *after* the pass (so the handle it reads is the
    pre-update state), and not at all during eval. Decorate the writer::

        @deferred_execution
        @torch.no_grad()
        def update(self, new_freqs) -> None:
            self.freqs.copy_(new_freqs)

    and call ``tracker.update(tracker.updated(acts))`` inline in the block."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        session = _ACTIVE_SESSION.get()
        if session is None:
            return method(self, *args, **kwargs)
        session.defer_call(method, self, args, kwargs)
        return None

    return wrapper


def bind[V](target: ValueRef[V], subs: Subs) -> ValueRef[V]:
    """Re-intern ``target``'s cone substituting producers/nodes per ``subs`` (a
    map from a slot-marker producer OR a node to its replacement). Because
    re-interning is global, two bindings producing the same structure share
    nodes. The value type ``V`` is preserved (re-interning rewrites structure, not
    the value a node yields).

    A producer may define ``__graph_rebind__(self, bound_inputs, subs)`` to
    control how its node re-interns -- the hook a sub-model slot uses to splice
    in a bound sub-graph (see ``spec.sub_slot``)."""
    memo: dict[ValueRef, ValueRef] = {}
    for n in _toposort(target):
        if n in subs:  # a value-slot node (e.g. Input) bound to another ref
            memo[n] = cast(ValueRef, subs[n])
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
    return cast("ValueRef[V]", memo[target])


def at[V](handle: ValueRef[V], x: ValueRef) -> ValueRef[V]:
    """Apply a sub-graph to a new input: rebind its ``Input`` to ``x`` -- so
    ``sae(src).acts`` is exactly ``at(sae.acts, src)``. Nothing is left unbound:
    ``bind`` rewrites *every* ``Input`` in ``handle``'s cone to ``x`` (an exhaustive
    substitution), preserving the value type. The graph-native "plug this sub-model
    in here" move -- e.g. running the SAE on a normalized input, or treating a
    mid-graph node as the input (``bind(recon, {acts: Input})``)."""
    return bind(handle, {Input: x})
