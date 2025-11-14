# # ---- VIRTUAL DIMS CORE -------------------------------------------------------
# from __future__ import annotations

# import operator
# from dataclasses import dataclass
# from typing import (
#     Any,
#     ClassVar,
#     get_args,
#     get_origin,
#     Iterable,
#     Iterator,
#     Optional,
#     Sequence,
#     Union,
# )

# import torch
# from torch import Tensor

# from saeco.data.dict_batch import DictBatch

# # Forward decls for type checkers
# if False:  # pragma: no cover
#     from typing_extensions import Self as _Self

# # ----------- Expression system (integers over dims) ---------------------------


# class DimExpr:
#     """Integer expression over dims and literals."""

#     __slots__ = ("op", "args", "require_exact_div")

#     def __init__(self, op: str, args: tuple[Any, ...], require_exact_div: bool = False):
#         self.op = op
#         self.args = args
#         self.require_exact_div = require_exact_div

#     def _bin(self, other, op, exact=False):
#         other = other if isinstance(other, (DimExpr, Dim, int)) else int(other)
#         return DimExpr(op, (self, other), exact)

#     def __add__(self, other):
#         return self._bin(other, "+")

#     def __radd__(self, other):
#         return DimExpr("+", (other, self))

#     def __sub__(self, other):
#         return self._bin(other, "-")

#     def __rsub__(self, other):
#         return DimExpr("-", (other, self))

#     def __mul__(self, other):
#         return self._bin(other, "*")

#     def __rmul__(self, other):
#         return DimExpr("*", (other, self))

#     def __floordiv__(self, other):
#         return self._bin(other, "//")

#     def __truediv__(self, other):
#         return self._bin(other, "/", exact=True)

#     def __mod__(self, other):
#         return self._bin(other, "%")

#     def __pow__(self, other):
#         return self._bin(other, "**")

#     def eval(self, state: "DimState") -> int:
#         def as_int(x):
#             if isinstance(x, int):
#                 return x
#             if isinstance(x, Dim):
#                 return state.get_len(x)
#             if isinstance(x, DimExpr):
#                 return x.eval(state)
#             raise TypeError(f"Unsupported expr arg: {x!r}")

#         if self.op == "lit":  # single literal
#             return as_int(self.args[0])

#         a = as_int(self.args[0])
#         b = as_int(self.args[1]) if len(self.args) > 1 else None

#         if self.op == "+":
#             return a + b
#         if self.op == "-":
#             return a - b
#         if self.op == "*":
#             return a * b
#         if self.op == "//":
#             return a // b
#         if self.op == "/":
#             if a % b != 0:
#                 raise ValueError(f"Expected exact division but {a}/{b} is not integer.")
#             return a // b
#         if self.op == "%":
#             return a % b
#         if self.op == "**":
#             return a**b
#         raise ValueError(f"Unknown op {self.op!r}")

#     @staticmethod
#     def lit(x: int) -> "DimExpr":
#         return DimExpr("lit", (int(x),))


# class DimRelation:
#     """Binary relation: ==, !=, <, <=, >, >= between Dim/DimExpr/int."""

#     __slots__ = ("lhs", "op", "rhs")

#     def __init__(
#         self, lhs: Union["Dim", DimExpr, int], op: str, rhs: Union["Dim", DimExpr, int]
#     ):
#         self.lhs, self.op, self.rhs = lhs, op, rhs

#     def check(self, state: "DimState") -> bool:
#         def val(x):
#             if isinstance(x, int):
#                 return x
#             if isinstance(x, Dim):
#                 return state.get_len(x)
#             if isinstance(x, DimExpr):
#                 return x.eval(state)
#             raise TypeError(x)

#         L, R = val(self.lhs), val(self.rhs)
#         ops = {
#             "==": operator.eq,
#             "!=": operator.ne,
#             "<": operator.lt,
#             "<=": operator.le,
#             ">": operator.gt,
#             ">=": operator.ge,
#         }
#         return ops[self.op](L, R)

#     def __repr__(self):
#         return f"{self.lhs} {self.op} {self.rhs}"


# # -------------------------- Dim & InstDim -------------------------------------


# class Dim:
#     """
#     A virtual dimension descriptor declared at class level:
#         class MyBatch(DictBatch):
#             B = Dim()
#             S = Dim()
#             T2 = Dim()
#             DIM_CONSTRAINTS = [ T2 == S*2 ]
#     """

#     __slots__ = ("name", "_fixed_len")

#     def __init__(self, l: Optional[int] = None):
#         self.name: Optional[str] = None
#         self._fixed_len = l  # if provided, this is a constant length (e.g., 1)

#     def __set_name__(self, owner, name):
#         self.name = name
#         reg = getattr(owner, "_registered_dims", None)
#         if reg is None:
#             reg = {}
#             setattr(owner, "_registered_dims", reg)
#         reg[name] = self

#     # Arithmetic → DimExpr
#     def _as_expr(self) -> DimExpr:
#         return DimExpr.lit(self)  # type: ignore[arg-type]

#     def __add__(self, other):
#         return self._as_expr().__add__(other)

#     def __radd__(self, other):
#         return other + self._as_expr()

#     def __sub__(self, other):
#         return self._as_expr().__sub__(other)

#     def __rsub__(self, other):
#         return other - self._as_expr()

#     def __mul__(self, other):
#         return self._as_expr().__mul__(other)

#     def __rmul__(self, other):
#         return other * self._as_expr()

#     def __truediv__(self, other):
#         return self._as_expr().__truediv__(other)  # exact division

#     def __floordiv__(self, other):
#         return self._as_expr().__floordiv__(other)

#     def __mod__(self, other):
#         return self._as_expr().__mod__(other)

#     def __pow__(self, other):
#         return self._as_expr().__pow__(other)

#     # Relations
#     def __eq__(self, other):
#         return DimRelation(self, "==", other)  # type: ignore[override]

#     def __ne__(self, other):
#         return DimRelation(self, "!=", other)  # type: ignore[override]

#     def __lt__(self, other):
#         return DimRelation(self, "<", other)

#     def __le__(self, other):
#         return DimRelation(self, "<=", other)

#     def __gt__(self, other):
#         return DimRelation(self, ">", other)

#     def __ge__(self, other):
#         return DimRelation(self, ">=", other)

#     # Pairing (for coupled indexing): B & BatchSizedDim
#     def __and__(self, other: "Dim"):
#         return PairedDims((self, other))

#     def __repr__(self):
#         return f"Dim({self.name})"


# class InstDim:
#     """Dim bound to a batch instance; supports indexing: batch.B[sel]."""

#     __slots__ = ("batch", "dim")

#     def __init__(self, batch: "DimAwareDictBatch", dim: Dim):
#         self.batch = batch
#         self.dim = dim

#     @property
#     def len(self) -> int:
#         return self.batch._dim_state.get_len(self.dim)

#     def __len__(self) -> int:
#         return self.len

#     def __getitem__(self, sel) -> "DimAwareDictBatch":
#         return self.batch._index_along({self.dim: sel})  # single-dim index


# # --------------------- Axis Spec (normal / broadcast / union) -----------------


# @dataclass(frozen=True)
# class AxisSpec:
#     kind: str  # "normal" | "broadcast" | "union"
#     dims: tuple[Dim, ...]  # 1 for normal/broadcast; 2+ for union

#     def __repr__(self):  # debug-friendly
#         if self.kind == "normal":
#             return f"{self.dims[0].name}"
#         if self.kind == "broadcast":
#             return f"[{self.dims[0].name}]"
#         return f"{'|'.join(d.name for d in self.dims)}"


# @dataclass
# class FieldShape:
#     axes: list[AxisSpec]  # per-tensor axis mapping


# # --------------------- Union & Pairing helpers --------------------------------


# class UnionDims:
#     """Represents B | S in annotations."""

#     __slots__ = ("options",)

#     def __init__(self, *dims: Dim):
#         self.options = dims


# class PairedDims:
#     """Represents B & BatchSizedDim for coupled indexing."""

#     __slots__ = ("dims",)

#     def __init__(self, dims: Sequence[Dim]):
#         self.dims = tuple(dims)


# def _union(a: Dim, b: Dim) -> UnionDims:
#     return UnionDims(a, b)


# # Monkeypatch "|" for union on Dim (PEP 604 style)
# Dim.__or__ = lambda self, other: _union(self, other)  # type: ignore[attr-defined]

# # --------------------- Dim state & registry -----------------------------------


# class DimState:
#     """Holds resolved lengths for all registered dims for one batch instance."""

#     def __init__(self, lengths: dict[Dim, int]):
#         self.lengths = lengths

#     def get_len(self, d: Dim) -> int:
#         if d not in self.lengths:
#             raise KeyError(f"Dim {d} has no resolved length")
#         return self.lengths[d]

#     def set_len(self, d: Dim, n: int):
#         if d in self.lengths and self.lengths[d] != n:
#             raise ValueError(
#                 f"Conflicting lengths for {d}: have {self.lengths[d]}, new {n}"
#             )
#         self.lengths[d] = int(n)


# # --------------------- Indexer descriptor -------------------------------------


# class Indexer:
#     """
#     Class-level descriptor:
#        by_seq = Indexer([S])
#        with_paired_B = Indexer([B & BatchSizedDim, S])

#     Usage:
#        batch.by_seq[slc]
#        batch.with_paired_B[idx]   # idx is applied to both B and BatchSizedDim
#     """

#     def __init__(self, groups: Sequence[Union[Dim, PairedDims]]):
#         self.groups = tuple(groups)
#         self.name: Optional[str] = None

#     def __set_name__(self, owner, name):
#         self.name = name

#     def __get__(self, obj, objtype=None):
#         if obj is None:
#             return self
#         return BoundIndexer(obj, self.groups)


# class BoundIndexer:
#     __slots__ = ("batch", "groups")

#     def __init__(self, batch: "DimAwareDictBatch", groups):
#         self.batch = batch
#         self.groups = groups

#     def __getitem__(self, sel):
#         """
#         sel can be:
#           - single index (applies to the only group)
#           - tuple of indices (one per group)
#         PairedDims consume one index for the *pair*.
#         """
#         if not isinstance(sel, tuple):
#             sel = (sel,)
#         if len(sel) != len(self.groups):
#             raise ValueError(
#                 f"Expected {len(self.groups)} index objects, got {len(sel)}"
#             )
#         mapping = {}
#         for group, s in zip(self.groups, sel):
#             if isinstance(group, PairedDims):
#                 for d in group.dims:
#                     mapping[d] = s
#             else:
#                 mapping[group] = s
#         return self.batch._index_along(mapping)


# # --------------------- Parsing field annotations ------------------------------


# def _is_dim(x) -> bool:
#     return isinstance(x, Dim)


# def _is_uniondims(x) -> bool:
#     return isinstance(x, UnionDims)


# def _parse_axis_spec(obj) -> AxisSpec:
#     # [[B]] / [B] is given by using a Sequence with length 1 and the element being a Dim.
#     # We treat a *list/tuple* containing exactly one Dim as broadcast axis.
#     if isinstance(obj, (list, tuple)):
#         if len(obj) != 1 or not _is_dim(obj[0]):
#             raise TypeError(f"Broadcast axis must be like [B], got {obj!r}")
#         return AxisSpec("broadcast", (obj[0],))
#     if _is_dim(obj):
#         return AxisSpec("normal", (obj,))
#     if _is_uniondims(obj):
#         return AxisSpec("union", tuple(obj.options))
#     raise TypeError(f"Unsupported axis spec: {obj!r}")


# def _parse_field_shape_spec(spec) -> FieldShape:
#     # Expect 'spec' to be a list/tuple of axis descriptors, each either:
#     #   Dim, [Dim] (list of one), or UnionDims (from B | S)
#     if not isinstance(spec, (list, tuple)):
#         raise TypeError(f"Field shape spec must be list/tuple, got {spec!r}")
#     axes = []
#     for axis in spec:
#         axes.append(_parse_axis_spec(axis))
#     return FieldShape(axes=axes)


# # -------------- Dim-aware DictBatch mixin -------------------------------------


# class DimAwareDictBatch:
#     """
#     Mixin that can be added to your DictBatch subclass to enable virtual dims.
#     It expects the subclass to define:
#         - some Dim descriptors (e.g., B = Dim())
#         - optional DIM_CONSTRAINTS: list[DimRelation]
#         - field -> shape spec mapping placed in __FIELD_SHAPES__ dict
#           e.g., __FIELD_SHAPES__ = {"tokens": [B, S], "mask": [[B], S], ...}
#     Or, if you prefer to drive via annotations, provide a small helper that
#     populates __FIELD_SHAPES__ at class decoration time (see notes below).
#     """

#     _registered_dims: ClassVar[dict[str, Dim]]
#     __FIELD_SHAPES__: ClassVar[dict[str, FieldShape]] = {}
#     DIM_CONSTRAINTS: ClassVar[list[DimRelation]] = []

#     DEFAULT_INDEXING_ORDERING: ClassVar[list[Union[Dim, Sequence[Dim]]]] = (
#         []
#     )  # tie-breaks unions

#     # Instance cache
#     _dim_state: DimState | None = None
#     _axis_map_per_field: dict[str, list[AxisSpec]] | None = None

#     def _ensure_dim_resolution(self):
#         if self._dim_state is not None:
#             return
#         # 1) gather all dims and initial fixed lengths
#         lengths: dict[Dim, int] = {}
#         for d in self._registered_dims.values():
#             if d._fixed_len is not None:
#                 lengths[d] = d._fixed_len
#         # 2) parse field shapes (already parsed in __FIELD_SHAPES__)
#         axis_map: dict[str, list[AxisSpec]] = {}
#         for k, v in self.items():
#             if k in self.__FIELD_SHAPES__:
#                 axis_map[k] = self.__FIELD_SHAPES__[k].axes
#             # else: tensor has no declared virtual dims → it will be unaffected by dim indexing
#         # 3) infer lengths from tensors
#         for field, axes in axis_map.items():
#             t: Tensor = self[field]
#             if t.ndim != len(axes):
#                 raise ValueError(
#                     f"Field {field!r} shape mismatch: tensor.ndim={t.ndim} vs spec={axes}"
#                 )
#             for ax_i, axspec in enumerate(axes):
#                 n = int(t.shape[ax_i])
#                 if axspec.kind == "normal":
#                     d = axspec.dims[0]
#                     _maybe_set_len(lengths, d, n, field, ax_i)
#                 elif axspec.kind == "broadcast":
#                     d = axspec.dims[0]
#                     if n != 1:
#                         raise ValueError(
#                             f"Field {field!r} axis {ax_i} marked broadcast [{d.name}] but length={n} != 1"
#                         )
#                     _maybe_set_len(
#                         lengths,
#                         d,
#                         lengths.get(d, None) or lengths.get(d, 1),
#                         field,
#                         ax_i,
#                     )
#                 else:  # union
#                     opts = axspec.dims
#                     candidates = [
#                         d for d in opts if d not in lengths or lengths[d] == n
#                     ]
#                     # prefer dims whose current or fixed len == n
#                     if len(candidates) == 1:
#                         _maybe_set_len(lengths, candidates[0], n, field, ax_i)
#                     elif len(candidates) > 1:
#                         # tie-break: DEFAULT_INDEXING_ORDERING order
#                         chosen = _tie_break_union(
#                             opts,
#                             n,
#                             lengths,
#                             getattr(self, "DEFAULT_INDEXING_ORDERING", []),
#                         )
#                         if chosen is None:
#                             raise ValueError(
#                                 f"Ambiguous union for field {field!r} axis {ax_i} with len={n}: options={opts}"
#                             )
#                         _maybe_set_len(lengths, chosen, n, field, ax_i)
#                     else:
#                         # none fit yet → if only one of opts has no assigned length, choose it
#                         unconstrained = [d for d in opts if d not in lengths]
#                         if len(unconstrained) == 1:
#                             _maybe_set_len(lengths, unconstrained[0], n, field, ax_i)
#                         else:
#                             # give up: ambiguous
#                             raise ValueError(
#                                 f"Cannot resolve union for field {field!r} axis {ax_i} len={n} options={opts}"
#                             )
#         state = DimState(lengths)

#         # 4) check constraints
#         for rel in getattr(self, "DIM_CONSTRAINTS", []):
#             if not rel.check(state):
#                 raise ValueError(f"Dim constraint failed: {rel}")

#         self._dim_state = state
#         self._axis_map_per_field = axis_map

#     # ---- public-ish helpers

#     def _inst_dim(self, d: Dim) -> InstDim:
#         self._ensure_dim_resolution()
#         return InstDim(self, d)

#     # called by InstDim and Indexer
#     def _index_along(self, mapping: dict[Dim, Any]) -> "DimAwareDictBatch":
#         self._ensure_dim_resolution()
#         # Apply selection per tensor according to its axis specs.
#         new_data: dict[str, Tensor] = {}
#         new_lengths = dict(self._dim_state.lengths)
#         for field, t in self.items():
#             axes = (self._axis_map_per_field or {}).get(field, None)
#             if axes is None:
#                 # Field not participating in virtual dims → unchanged
#                 new_data[field] = t
#                 continue
#             new_t = t
#             # Build per-axis selectors, starting with default ":" for all
#             selectors: list[Any] = [slice(None)] * t.ndim
#             for ax_i, axspec in enumerate(axes):
#                 if axspec.kind == "normal":
#                     d = axspec.dims[0]
#                     if d in mapping:
#                         selectors[ax_i] = mapping[d]
#                 elif axspec.kind == "broadcast":
#                     d = axspec.dims[0]
#                     if d in mapping:
#                         # axis is length-1 but participates in dim d
#                         # expand to new length if a Tensor index (mask/indices) or slice with step resolves length
#                         sel = mapping[d]
#                         sel_len = _sel_length(sel, new_lengths[d])
#                         # make sure axis is length-1 now, then expand before indexing
#                         new_t = new_t.expand(
#                             *(
#                                 new_t.shape[:ax_i]
#                                 + (new_lengths[d],)
#                                 + new_t.shape[ax_i + 1 :]
#                             )
#                         )
#                         selectors[ax_i] = sel
#                         new_lengths[d] = sel_len
#                 else:  # union
#                     # figure out which dim this axis actually resolved to
#                     d = _resolved_union_dim_for_axis(axspec, self._dim_state)
#                     if d in mapping:
#                         selectors[ax_i] = mapping[d]
#                         new_lengths[d] = _sel_length(mapping[d], new_lengths[d])

#             new_t = new_t[tuple(selectors)]
#             new_data[field] = new_t

#         # Update and rewrap
#         new_batch = self.__class__.construct_with_other_data(new_data, self._get_other_dict())  # type: ignore[attr-defined]
#         # Reuse dim maps but update state (re-check constraints)
#         new_batch._dim_state = DimState(new_lengths)
#         new_batch._axis_map_per_field = self._axis_map_per_field
#         for rel in getattr(self, "DIM_CONSTRAINTS", []):
#             if not rel.check(new_batch._dim_state):
#                 raise ValueError(f"Dim constraint failed after indexing: {rel}")
#         return new_batch


# # ---- small helpers -----------------------------------------------------------


# def _maybe_set_len(lengths: dict[Dim, int], d: Dim, n: int, field: str, ax_i: int):
#     if d in lengths and lengths[d] not in (n, 1):
#         raise ValueError(
#             f"Conflicting length for dim {d.name}: seen {lengths[d]} vs {n} at {field}[axis {ax_i}]"
#         )
#     lengths[d] = int(n)


# def _tie_break_union(
#     options: Sequence[Dim],
#     n: int,
#     lengths: dict[Dim, int],
#     ordering: Sequence[Union[Dim, Sequence[Dim]]],
# ) -> Optional[Dim]:
#     # Prefer dims earlier in DEFAULT_INDEXING_ORDERING whose current or fixed len is n or unset.
#     def candidates():
#         for item in ordering:
#             if isinstance(item, Dim):
#                 if item in options and (item not in lengths or lengths[item] == n):
#                     yield item
#             else:  # sequence of dims
#                 for d in item:
#                     if d in options and (d not in lengths or lengths[d] == n):
#                         yield d

#     try:
#         return next(candidates())
#     except StopIteration:
#         # If none from ordering, any option with n or unset
#         for d in options:
#             if d not in lengths or lengths[d] == n:
#                 return d
#         return None


# def _resolved_union_dim_for_axis(axspec: AxisSpec, state: DimState) -> Dim:
#     assert axspec.kind == "union"
#     # The chosen one is the option whose length equals the axis length in the tensor.
#     # We don’t have the axis length here directly; we rely on the fact that at inference time
#     # we set exactly one of these options to the actual axis length in state.
#     # If multiple options share equal length, fall back to first present in state (stable by resolve).
#     present = [d for d in axspec.dims if d in state.lengths]
#     if not present:
#         raise ValueError(f"Union axis has no resolved dim.")
#     if len(present) == 1:
#         return present[0]
#     # If there are multiple equal lengths, use first
#     return present[0]


# def _sel_length(sel, current_len: int) -> int:
#     # Best effort determination of resulting length after indexing:
#     # - slice → compute range
#     # - Tensor[bool] mask → mask.sum()
#     # - Tensor[int] indices → indices.numel() or size along last?
#     # - int → 1 (we keep rank; if you want rank-drop, add an option)
#     if isinstance(sel, slice):
#         start, stop, step = sel.start, sel.stop, sel.step
#         step = 1 if step is None else step
#         start = 0 if start is None else (start if start >= 0 else current_len + start)
#         stop = (
#             current_len if stop is None else (stop if stop >= 0 else current_len + stop)
#         )
#         # clamp
#         start = max(0, min(current_len, start))
#         stop = max(0, min(current_len, stop))
#         if stop <= start:
#             return 0
#         return (stop - start + (step - 1)) // step
#     if isinstance(sel, int):
#         return 1
#     if isinstance(sel, Tensor):
#         if sel.dtype == torch.bool:
#             return int(sel.sum().item())
#         else:
#             return int(sel.numel())
#     if isinstance(sel, (list, tuple)):
#         return len(sel)
#     # default fallback
#     return current_len


# class DimBatch(DimAwareDictBatch, DictBatch):
#     """
#     Use this instead of DictBatch for batches with virtual dims.
#     You can keep DictBatch for non-dim-aware batches.
#     """

#     # Convenience: expose dims as instance properties
#     def __getattr__(self, name):
#         # first: let DictBatch try TENSOR_DATA_FIELDS etc.
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             pass
#         # then dims:
#         dims = getattr(self.__class__, "_registered_dims", {})
#         if name in dims:
#             return self._inst_dim(dims[name])
#         raise
# =========================
# Virtual Dimension System
# =========================
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    get_args,
    get_origin,
    Iterable,
    Iterator,
    Mapping,
    overload,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import Tensor

from saeco.data.dict_batch.dict_batch import DictBatch


# ---------- Dim descriptors ----------


class Dim:
    """
    Class-level descriptor that registers a named logical axis.
    Use `Dim(l=1)` to declare a constant-length dim (e.g., singular).
    """

    __slots__ = ("name", "_owner", "_const_len")

    def __init__(self, l: int | None = None):
        self.name: str | None = None
        self._owner: type[DictBatch] | None = None
        self._const_len = l

    def __set_name__(self, owner, name):
        self.name = name
        self._owner = owner
        reg = getattr(owner, "_registered_dims", None)
        if reg is None:
            owner._registered_dims = {}
        owner._registered_dims[name] = self  # type: ignore[attr-defined]

    # descriptor: instance → InstDim; class → Dim itself
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return InstDim(obj, self)

    # unions/pairs as syntax sugar in annotations/indexers
    def __or__(self, other: Dim) -> DimUnion:
        return DimUnion((self, other))

    # arithmetic expressions for constraints
    def __mul__(self, k: int | Expr) -> Expr:
        return Mul(Var(self), ensure_expr(k))

    def __rmul__(self, k: int | Expr) -> Expr:
        return Mul(ensure_expr(k), Var(self))

    def __add__(self, k: int | Expr) -> Expr:
        return Add(Var(self), ensure_expr(k))

    def __radd__(self, k: int | Expr) -> Expr:
        return Add(ensure_expr(k), Var(self))

    def __truediv__(self, k: int | Expr) -> Expr:
        return Div(Var(self), ensure_expr(k), require_int=True)

    def __floordiv__(self, k: int | Expr) -> Expr:
        return Div(Var(self), ensure_expr(k), floor=True)

    def __pow__(self, k: int | Expr) -> Expr:
        return Pow(Var(self), ensure_expr(k))

    def __repr__(self):
        return f"<Dim {self.name or '?'}>"


class DimUnion:
    __slots__ = ("options",)

    def __init__(self, options: Tuple[Dim, ...] | Sequence[Dim]):
        opts = tuple(options)
        if len(opts) < 2:
            raise ValueError("DimUnion requires >=2 dims")
        self.options: Tuple[Dim, ...] = opts

    def __or__(self, other: Dim) -> DimUnion:
        if isinstance(other, Dim):
            return DimUnion(self.options + (other,))
        raise TypeError("Union only supports Dim")

    def __repr__(self):
        names = " | ".join(d.name or "?" for d in self.options)
        return f"({names})"


@dataclass(frozen=True)
class AxisSpec:
    """One physical axis spec for a tensor field."""

    kind: str  # "axis" | "broadcast" | "union"
    dim: Dim | None = None
    union: Tuple[Dim, ...] | None = None

    @staticmethod
    def axis(d: Dim) -> "AxisSpec":
        return AxisSpec("axis", dim=d)

    @staticmethod
    def broadcast(d: Dim) -> "AxisSpec":
        return AxisSpec("broadcast", dim=d)

    @staticmethod
    def union(ds: Sequence[Dim]) -> "AxisSpec":
        return AxisSpec("union", union=tuple(ds))


# ---------- Expressions & constraints (simple) ----------


class Expr:  # base
    def eval(self, env: Mapping[Dim, int]) -> int | None:
        raise NotImplementedError


@dataclass
class Const(Expr):
    v: int

    def eval(self, env):
        return int(self.v)


@dataclass
class Var(Expr):
    d: Dim

    def eval(self, env):
        return env.get(self.d, self.d._const_len)


@dataclass
class Add(Expr):
    a: Expr
    b: Expr

    def eval(self, env):
        av, bv = self.a.eval(env), self.b.eval(env)
        return None if (av is None or bv is None) else av + bv


@dataclass
class Mul(Expr):
    a: Expr
    b: Expr

    def eval(self, env):
        av, bv = self.a.eval(env), self.b.eval(env)
        return None if (av is None or bv is None) else av * bv


@dataclass
class Div(Expr):
    a: Expr
    b: Expr
    require_int: bool = False
    floor: bool = False

    def eval(self, env):
        av, bv = self.a.eval(env), self.b.eval(env)
        if av is None or bv is None:
            return None
        if bv == 0:
            raise ZeroDivisionError("dim expr division by zero")
        if self.floor:
            return av // bv
        if self.require_int and av % bv != 0:
            raise ValueError(f"non-integer dim division: {av}/{bv}")
        return av // bv if self.require_int else av / bv  # normally keep int by inputs


@dataclass
class Pow(Expr):
    a: Expr
    b: Expr

    def eval(self, env):
        av, bv = self.a.eval(env), self.b.eval(env)
        return None if (av is None or bv is None) else int(av**bv)


def ensure_expr(x: int | Expr) -> Expr:
    return x if isinstance(x, Expr) else Const(int(x))


@dataclass
class Constraint:
    """left must be a single Dim; right is an Expr."""

    left: Dim
    right: Expr

    def try_solve_into(self, env: Dict[Dim, int]) -> bool:
        if self.left in env:
            return False
        # compute RHS if possible
        rv = self.right.eval(env)
        if rv is None:
            return False
        if rv < 0:
            raise ValueError(f"Negative length for {self.left}: {rv}")
        env[self.left] = int(rv)
        return True


# ---------- Bound dims (sugar for isel) ----------


class InstDim:
    __slots__ = ("inst", "dim")

    def __init__(self, inst: DictBatch, dim: Dim):
        self.inst = inst
        self.dim = dim

    def __and__(self, other: "InstDim | Dim") -> "PairedInstDims":
        other_dim = other.dim if isinstance(other, InstDim) else other
        return PairedInstDims(self.inst, (self.dim, other_dim))

    def __getitem__(self, indexer: Any):
        return self.inst.isel(**{self.dim: indexer})

    def __len__(self):
        return self.inst._dim_env()[self.dim]

    def __repr__(self):
        return f"{self.inst.__class__.__name__}.{self.dim.name}"


class PairedInstDims:
    __slots__ = ("inst", "dims")

    def __init__(self, inst: DictBatch, dims: Tuple[Dim, Dim]):
        self.inst = inst
        self.dims = dims

    def __getitem__(self, indexer: Any):
        d1, d2 = self.dims
        return self.inst.isel(**{d1: indexer, d2: indexer})


# ---------- Utilities: parsing specs from annotations ----------


def _parse_shape_spec_from_hint(hint, cls) -> list[AxisSpec] | None:
    """
    Understand annotations like Int[Tensor, [B, S]],
    Int[Tensor, [[B], S]] (broadcast B), or Int[Tensor, [B | S]] (union).
    Returns None if we can't find a shape spec (fallback later).
    """
    try:
        origin = get_origin(hint)
        args = get_args(hint)
        # jaxtyping-like: Int[Tensor, <shape>]
        if args and len(args) >= 2:
            shape = args[1]
            return _parse_shape_list(shape, cls)
    except Exception:
        pass
    return None


def _parse_shape_list(shape, cls) -> list[AxisSpec]:
    axes: list[AxisSpec] = []

    def _is_dim(x) -> bool:
        return isinstance(x, Dim) or (isinstance(x, InstDim))  # generous

    # shape might be a list like [B, S] or [[B], S] or include unions B|S
    for ax in list(shape):
        # broadcast: a single-element list like [B]
        if isinstance(ax, (list, tuple)) and len(ax) == 1 and _is_dim(ax[0]):
            d = ax[0].dim if isinstance(ax[0], InstDim) else ax[0]
            axes.append(AxisSpec.broadcast(d))
        # union via DimUnion
        elif isinstance(ax, DimUnion):
            axes.append(AxisSpec.union(ax.options))
        elif _is_dim(ax):
            d = ax.dim if isinstance(ax, InstDim) else ax
            axes.append(AxisSpec.axis(d))
        else:
            raise TypeError(f"Unrecognized axis spec element: {ax!r}")
    return axes


# ============= DictBatch mixin features =============


def _axis_for_dim(
    specs: list[AxisSpec], dim: Dim, env: Mapping[Dim, int]
) -> int | None:
    for i, ax in enumerate(specs):
        if ax.kind == "axis" and ax.dim is dim:
            return i
        if ax.kind == "broadcast" and ax.dim is dim:
            return i
        if ax.kind == "union" and dim in ax.union:
            # choose the union branch that matches env
            # if multiple could match, prefer first by declaration
            # (you can get fancier by comparing actual size)
            return i
    return None


# Public decorator to enable dims on a DictBatch subclass
def enable_virtual_dims(cls: type[DictBatch]) -> type[DictBatch]:
    # Gather per-field axis specs
    cls._FIELD_DIMS: Dict[str, list[AxisSpec]] = {}
    hints = getattr(cls, "__annotations__", {})
    for name, hint in hints.items():
        if name in getattr(cls, "TENSOR_DATA_FIELDS", ()) or name in getattr(
            cls, "__dict__", {}
        ):
            specs = _parse_shape_spec_from_hint(hint, cls)
            if specs is not None:
                cls._FIELD_DIMS[name] = specs

    # Gather constraints
    raw = getattr(cls, "DIM_CONSTRAINTS", [])
    constraints: list[Constraint] = []
    for c in raw:
        # Expect forms like:  SeqTimes2Dim == (S * 2)
        if not (isinstance(c, tuple) and len(c) == 2 and c[0] == "eq"):
            # allow direct Constraint
            if isinstance(c, Constraint):
                constraints.append(c)
            else:
                raise TypeError(
                    "Each DIM_CONSTRAINTS entry must be Constraint(...) or ('eq', Dim, Expr)."
                )
        else:
            _, left, right = c
            if not isinstance(left, Dim):
                raise TypeError("Constraint left must be a Dim")
            if not isinstance(right, Expr):
                right = ensure_expr(right)
            constraints.append(Constraint(left, right))
    cls._DIM_CONSTRAINTS = constraints

    # Attach helpers to the class
    def _dim_env(self: DictBatch) -> Dict[Dim, int]:
        # cache lazily per-instance
        env: Dict[Dim, int] = {}
        # 1) fill from const dims
        for d in getattr(self.__class__, "_registered_dims", {}).values():
            if d._const_len is not None:
                env[d] = int(d._const_len)

        # 2) infer from tensor shapes
        field_dims = getattr(self.__class__, "_FIELD_DIMS", {})
        for key, specs in field_dims.items():
            if key not in self:  # skip absent fields
                continue
            t = self[key]
            if not isinstance(t, Tensor):
                continue
            if t.dim() != len(specs):
                raise ValueError(
                    f"Field {key!r} has rank {t.dim()} but {len(specs)} axes are declared"
                )
            for i, ax in enumerate(specs):
                if ax.kind == "axis" and ax.dim not in env:
                    env[ax.dim] = int(t.shape[i])
                elif ax.kind == "broadcast":
                    # must be physical size 1; fill the target dim if known later
                    if t.shape[i] != 1:
                        raise ValueError(
                            f"Field {key!r} broadcast dim {ax.dim} must have size 1, got {t.shape[i]}"
                        )
                elif ax.kind == "union":
                    # choose the union member whose currently-known size matches; otherwise defer
                    known = [d for d in ax.union if d in env]
                    if known:
                        # sanity check: if we know one, its size should match
                        if t.shape[i] != env[known[0]]:
                            # defer: might be another member; don't fail yet
                            pass
                    else:
                        # if none known, try to set exactly one by size matching
                        matches = [
                            d
                            for d in ax.union
                            if (d._const_len or t.shape[i])
                            == (d._const_len or t.shape[i])
                        ]
                        # naive: if only one candidate or constants agree, set it
                        # (you can make this smarter: try all)
                        # don't write to env here; union resolves when other fields force it
                        pass

        # 3) solve equality constraints left == expr(env)
        changed = True
        iters = 0
        constraints = getattr(self.__class__, "_DIM_CONSTRAINTS", [])
        while changed and iters < 8:
            changed, iters = False, iters + 1
            for c in constraints:
                changed = c.try_solve_into(env) or changed

        # 4) sanity: all dims referenced in specs should be known by now
        # (You may make this permissive and delay to validate_dims)
        return env

    def validate_dims(self: DictBatch):
        env = self._dim_env()
        # verify all axis lengths consistent with env (+ broadcast/union OK)
        field_dims = getattr(self.__class__, "_FIELD_DIMS", {})
        for key, specs in field_dims.items():
            if key not in self:
                continue
            t = self[key]
            for i, ax in enumerate(specs):
                if ax.kind == "axis":
                    want = env.get(ax.dim, ax.dim._const_len)
                    if want is None:
                        continue
                    if t.shape[i] != want:
                        raise ValueError(
                            f"{key}: axis {i} for {ax.dim} mismatches: {t.shape[i]} != {want}"
                        )
                elif ax.kind == "broadcast":
                    if t.shape[i] != 1:
                        raise ValueError(
                            f"{key}: broadcast axis {i} must be 1, got {t.shape[i]}"
                        )
                elif ax.kind == "union":
                    # axis length must equal at least one candidate in env (if any are known)
                    known = [(d, env[d]) for d in ax.union if d in env]
                    if known and all(t.shape[i] != v for _, v in known):
                        raise ValueError(
                            f"{key}: union axis {i} length {t.shape[i]} not in {[v for _, v in known]}"
                        )

    def axis_of(self: DictBatch, field: str, dim: Dim) -> int:
        specs = self.__class__._FIELD_DIMS.get(field)
        if specs is None:
            raise KeyError(f"No shape spec for field {field!r}")
        idx = _axis_for_dim(specs, dim, self._dim_env())
        if idx is None:
            raise ValueError(f"Field {field!r} has no axis for dim {dim}")
        return idx

    # ---- named-dim indexing ----
    def isel(self: DictBatch, **indexers: Mapping[Dim | str, Any]):
        """
        Index by named dims: batch.isel(B=idx, S=slice(None))
        Accept Dim instances or their names (str).
        """
        # normalize keys (Dim or str) -> Dim
        key2dim: Dict[Dim, Any] = {}
        for k, v in indexers.items():
            if isinstance(k, Dim):
                key2dim[k] = v
            elif isinstance(k, str):
                d = self.__class__._registered_dims.get(k)
                if d is None:
                    raise KeyError(f"Unknown dim name {k!r}")
                key2dim[d] = v
            else:
                raise TypeError("isel keys must be Dim or str")

        env = self._dim_env()
        out_data: Dict[str, Tensor] = {}
        for field, t in self.items():
            specs = self.__class__._FIELD_DIMS.get(field)
            if specs is None:
                # legacy field: apply only if first axis is selected via B
                # else passthrough
                out_data[field] = t
                continue

            # prepare per-axis index; expand broadcast dims if selected
            idx_tuple: list[Any] = [slice(None)] * t.dim()
            expanded_t = t
            for d, idx in key2dim.items():
                a = _axis_for_dim(specs, d, env)
                if a is None:
                    continue  # dim not present in this tensor
                axspec = specs[a]
                if axspec.kind == "broadcast":
                    want = env[d]
                    if expanded_t.shape[a] != want:
                        # expand view
                        ex_shape = list(expanded_t.shape)
                        ex_shape[a] = want
                        expanded_t = expanded_t.expand(*ex_shape)
                idx_tuple[a] = idx

            out_data[field] = expanded_t.__getitem__(tuple(idx_tuple))

        return self.__class__.construct_with_other_data(
            out_data, self._get_other_dict()
        )

    # ---- named-dim gather ----
    def gather_dim(self: DictBatch, dim: Dim | str, indices: Tensor) -> DictBatch:
        d = dim if isinstance(dim, Dim) else self.__class__._registered_dims[dim]
        env = self._dim_env()
        out_data: Dict[str, Tensor] = {}
        for field, t in self.items():
            specs = self.__class__._FIELD_DIMS.get(field)
            if specs is None:
                out_data[field] = t  # or raise
                continue
            a = _axis_for_dim(specs, d, env)
            if a is None:
                out_data[field] = t
                continue
            axspec = specs[a]
            tt = t
            if axspec.kind == "broadcast":
                want = env[d]
                if tt.shape[a] != want:
                    ex_shape = list(tt.shape)
                    ex_shape[a] = want
                    tt = tt.expand(*ex_shape)
            out_data[field] = tt.gather(a, indices)
        return self.__class__.construct_with_other_data(
            out_data, self._get_other_dict()
        )

    # ---- cat/stack on named dims ----
    @classmethod
    def cat_named(cls, batches: list["DictBatch"], dim: Dim | str) -> "DictBatch":
        d = dim if isinstance(dim, Dim) else cls._registered_dims[dim]
        cls._validate_keysets(batches)
        envs = [b._dim_env() for b in batches]
        out: Dict[str, Tensor] = {}
        for key in batches[0].keys():
            specs = getattr(cls, "_FIELD_DIMS", {}).get(key)
            if specs is None:
                # fallback: cat on axis 0
                out[key] = torch.cat([b[key] for b in batches], dim=0)
                continue
            # find axis
            a = _axis_for_dim(specs, d, envs[0])
            if a is None:
                # must be identical across batches
                out[key] = batches[0][key]
                continue
            # make each tensor compatible (broadcast axes expanded)
            to_cat: list[Tensor] = []
            for b, env in zip(batches, envs):
                t = b[key]
                # expand broadcast dims to env lengths
                tt = t
                for i, ax in enumerate(specs):
                    if ax.kind == "broadcast":
                        want = env[ax.dim]
                        if tt.shape[i] != want:
                            ex_shape = list(tt.shape)
                            ex_shape[i] = want
                            tt = tt.expand(*ex_shape)
                to_cat.append(tt)
            out[key] = torch.cat(to_cat, dim=a)
        # merge extras using your existing policy
        return cls.construct_with_other_data(out, cls._merge_other_data(batches))

    @classmethod
    def stack_named(cls, batches: list["DictBatch"], dim: Dim | str) -> "DictBatch":
        d = dim if isinstance(dim, Dim) else cls._registered_dims[dim]
        cls._validate_keysets(batches)
        env = batches[0]._dim_env()
        out: Dict[str, Tensor] = {}
        for key in batches[0].keys():
            specs = getattr(cls, "_FIELD_DIMS", {}).get(key)
            if specs is None:
                out[key] = torch.stack([b[key] for b in batches], dim=0)
                continue
            a = _axis_for_dim(specs, d, env)
            if a is None:
                out[key] = torch.stack([b[key] for b in batches], dim=0)
                continue
            # move the named axis so we can stack new leading axis and then move back, or simpler:
            # just stack along a new axis and then permute if you want exact placement; for brevity:
            out[key] = torch.stack([b[key] for b in batches], dim=a)
        return cls.construct_with_other_data(out, cls._merge_other_data(batches))

    # bind methods
    cls._dim_env = _dim_env
    cls.validate_dims = validate_dims
    cls.axis_of = axis_of
    cls.isel = isel
    cls.gather_dim = gather_dim
    cls.cat_named = cat_named
    cls.stack_named = stack_named
    return cls


from jaxtyping import Int


@DictBatch.auto_other_fields
@enable_virtual_dims
class SeqDatasExampleDimsUsage(DictBatch):
    # declare dims
    B: Dim = Dim()
    S: Dim = Dim()
    SeqTimes2Dim: Dim = Dim()
    SingularDim: Dim = Dim(l=1)
    BatchSizedDim: Dim = Dim()

    # constraints
    DIM_CONSTRAINTS: ClassVar[list[Any]] = [
        SeqTimes2Dim == S * 2,  # SeqTimes2Dim == 2 * S
        BatchSizedDim == B,  # BatchSizedDim == B
    ]

    # tensors (shape specs parsed from jaxtyping-ish hints)
    tokens: Int[Tensor, [B, S]]
    batch_len_only: Int[Tensor, [B]]
    unsqueezed: Int[Tensor, [[B], SingularDim]]  # physical (1, 1); both logical
    squeezed_batch_position: Int[Tensor, [[B], S]]
    batch_square_independent: Int[Tensor, [B, BatchSizedDim]]
    batch_or_seq: Int[Tensor, [B | S]]


# Build a batch and play
batch = SeqDatasExampleDimsUsage(
    tokens=torch.arange(12).view(3, 4),
    batch_len_only=torch.ones(3, dtype=torch.long),
    unsqueezed=torch.ones(1, 1, dtype=torch.long),
    squeezed_batch_position=torch.arange(4).view(1, 4),
    batch_square_independent=torch.arange(9).view(3, 3),
    batch_or_seq=torch.arange(3),  # matches B
)

batch.validate_dims()  # raises if anything is off

# Named-dim indexing
i = torch.tensor([0, 2])
b2 = batch.B[i]  # == batch.isel(B=i)
b3 = batch.isel(S=slice(1, None))

# Zipped/paired indexing (diagonal on (B, BatchSizedDim))
b_diag = (batch.B & batch.BatchSizedDim)[i]

# Gather along named dim (broadcast-aware)
idx = torch.tensor([[0, 2], [1, 1]])  # arbitrary shape
b_g = batch.gather_dim("B", idx)

# Concatenate along a named dim
out = SeqDatasExampleDimsUsage.cat_named([batch, batch], dim="B")

# sketching


# class InstDim[T: DictBatch]:
#     inst: T
#     dim_name: str
#     dim: Dim[T]

#     @property
#     def len(self): ...

#     def __getitem__(self, key: str) -> T: ...


# class Dim[T: DictBatch]:
#     ### this would be a property like object I think
#     def __get__(self, obj, objtype=None) -> InstDim[T]: ...
#     def __set_name__(self, owner, name) -> None:
#         # modify the owner so it knows it has a dim of this type
#         owner._registered_dims[name] = self  # something like this?
#         ...

#     def __mul__(self, other: DimRelation) -> DimRelation: ...
#     def __truediv__(
#         self, other: DimRelation
#     ) -> (
#         DimRelation
#     ): ...  # this one should make sure that for any final tensors the dim length result is an integer
#     def __floordiv__(
#         self, other: DimRelation
#     ) -> DimRelation: ...  # whereas this one will round down to the nearest integer
#     def __mod__(self, other: DimRelation) -> DimRelation: ...
#     def __pow__(self, other: DimRelation) -> DimRelation: ...
#     def __len__(self) -> int: ...
#     def __eq__(self, other: DimRelation) -> bool: ...
#     def __ne__(self, other: DimRelation) -> bool: ...
#     def __gt__(self, other: DimRelation) -> bool: ...
#     def __ge__(self, other: DimRelation) -> bool: ...
#     def __lt__(self, other: DimRelation) -> bool: ...
#     def __le__(self, other: DimRelation) -> bool: ...
#     def __radd__(self, other: DimRelation) -> DimRelation: ...
#     def __rsub__(self, other: DimRelation) -> DimRelation: ...


# from typing import Sequence

# from jaxtyping import Float, Int


# @DictBatch.auto_other_fields
# class SeqDatasExampleDimsUsage(DictBatch):
#     B: Dim = Dim()
#     S: Dim = Dim()
#     SeqTimes2Dim = Dim()
#     SingularDim: Dim = Dim(l=1)
#     BatchSizedDim: Dim = Dim()
#     DIM_CONSTRAINTS: ClassVar[list[DimRelation]] = [
#         SeqTimes2Dim == S * 2,
#         BatchSizedDim == B,
#     ]
#     tokens: Int[Tensor, [B, S]]
#     batch_len_only: Int[Tensor, [B]]
#     unsqueezed: Int[Tensor, [B, SingularDim]]  # this will be a (B,1)
#     squeezed_batch_position: Int[
#         Tensor, [[B], S]
#     ]  # this will be a (1,S) tensor and will be positioned like dim B for access and indexing
#     # the [] around B denotes that it's a singular/broadcast dim
#     # it will be treated like it broadcasts to B
#     batch_square_independent: Int[
#         Tensor, [B, BatchSizedDim]
#     ]  # this will be a (B,B) tensor, but position 1 (BatchSizedDim) is indexed separately dim from B

#     DEFAULT_INDEXING_ORDERING: ClassVar[list[Dim | Sequence[Dim]]] = [
#         B,
#         S,
#         SingularDim,
#         BatchSizedDim,
#     ]  # this maybe implicitly defines an Indexer?
#     # also can explicitly define indexers:
#     by_seq = Indexer([S])
#     with_paired_B = Indexer([B & BatchSizedDim, S, SingularDim])


# class InstDim[T: DictBatch]:
#     inst: T
#     dim_name: str
#     dim: Dim[T]

#     @property
#     def len(self): ...

#     def __getitem__(self, key: str) -> T: ...


# class Dim[T: DictBatch]:
#     ### this would be a property like object I think
#     def __get__(self, obj, objtype=None) -> InstDim[T]: ...
#     def __set_name__(self, owner, name) -> None:
#         # modify the owner so it knows it has a dim of this type
#         owner._registered_dims[name] = self  # something like this?
#         ...


# from typing import Sequence

# from jaxtyping import Float, Int


# @DictBatch.auto_other_fields
# class SeqDatasExampleDimsUsage(DictBatch):
#     B: Dim = Dim()
#     S: Dim = Dim()
#     SingularDim: Dim = Dim(l=1)
#     BatchSizedDim: Dim = Dim(eq=B)
#     tokens: Int[Tensor, [B, S]]
#     batch_len_only: Int[Tensor, [B]]
#     unsqueezed: Int[Tensor, [B, SingularDim]]  # this will be a (B,1)
#     squeezed_batch_position: Int[
#         Tensor, [[B], S]
#     ]  # this will be a (1,S) tensor and will be positioned like dim B for access and indexing
#     # the [] around B denotes that it's a singular/broadcast dim
#     # it will be treated like it broadcasts to B
#     batch_square_independent: Int[
#         Tensor, [B, BatchSizedDim]
#     ]  # this will be a (B,B) tensor, but position 1 (BatchSizedDim) is indexed separately dim from B

#     DEFAULT_INDEXING_ORDERING: ClassVar[list[Dim | Sequence[Dim]]] = [
#         B,
#         S,
#         SingularDim,
#         BatchSizedDim,
#     ]  # this maybe implicitly defines an Indexer?
#     # also can explicitly define indexers:
#     by_seq = Indexer([S])
#     with_paired_B = Indexer([B & BatchSizedDim, S, SingularDim])
