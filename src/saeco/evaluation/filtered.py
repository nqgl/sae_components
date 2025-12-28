from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Literal, Self, overload

import torch
from attrs import define, field
from torch import Tensor

from saeco.data.dict_batch import DictBatch

from .named_filter import NamedFilter

type Indexable = Tensor | DictBatch
type Slices = tuple[slice | int | None, ...]


def _is_full_slice(s: slice) -> bool:
    return s.start is None and s.stop is None and s.step is None


def _normalize_slices(slices: Sequence[slice | int | None] | None) -> Slices:
    if slices is None:
        return ()
    out: Slices = tuple(slices)
    while out and (
        out[-1] is None or (isinstance(out[-1], slice) and _is_full_slice(out[-1]))
    ):
        out = out[:-1]
    return out


def _as_index_tuple(slices: Slices) -> tuple[slice | int, ...]:
    # IMPORTANT: In this codebase, "None" means "no-op / full slice", NOT numpy-newaxis.
    return tuple(slice(None) if s is None else s for s in slices)


def _normalize_slice(s: slice, dim: int) -> tuple[int, int, int]:
    step = 1 if s.step is None else s.step
    if step <= 0:
        raise ValueError(f"Only positive slice steps are supported, got step={step}")

    start = 0 if s.start is None else s.start
    stop = dim if s.stop is None else s.stop

    # Handle negative indices similar to Python slicing.
    if start < 0:
        start += dim
    if stop < 0:
        stop += dim

    start = max(0, min(start, dim))
    stop = max(0, min(stop, dim))
    return start, stop, step


def _slice_len(dim: int, s: slice | int | None) -> int:
    match s:
        case None:
            return dim
        case int():
            return 1
        case slice() as sl:
            start, stop, step = _normalize_slice(sl, dim)
            if stop <= start:
                return 0
            return (stop - start + step - 1) // step
    raise TypeError(f"Unexpected slice spec: {type(s)}")


def _apply_slices(value: Indexable, slices: Slices) -> Indexable:
    if not slices:
        return value

    idx = _as_index_tuple(slices)

    if isinstance(value, DictBatch):
        # DictBatch only supports slicing on batch dimension (dim 0).
        if len(idx) > 1:
            non_batch = [
                s for s in idx[1:] if not (isinstance(s, slice) and _is_full_slice(s))
            ]
            if non_batch:
                raise ValueError(
                    f"DictBatch only supports dim0 slicing; got non-trivial slices beyond dim0: {idx[1:]}"
                )
        sl0 = idx[0] if idx else slice(None)
        return value[sl0]

    # Tensor: standard indexing, trailing dims are implicitly full slices.
    return value[idx]


def _sparse_row_mask(v: Tensor, mask: Tensor) -> Tensor:
    """
    Apply a 1D boolean row-mask to a sparse COO tensor.

    Result is re-indexed so masked rows become [0..mask.sum()).
    """
    if not v.is_sparse:
        raise TypeError("_sparse_row_mask expects a sparse COO tensor")

    v = v.coalesce()
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got {mask.ndim}D")

    idx = v.indices()
    vals = v.values()
    rows = idx[0]

    mask = mask.to(device=rows.device)
    keep = mask[rows]

    # Map old row indices -> new row indices in masked space.
    n_selected = int(mask.sum().item())
    inv = torch.full((mask.numel(),), -1, dtype=torch.long, device=rows.device)
    inv[mask] = torch.arange(n_selected, device=rows.device)

    new_rows = inv[rows[keep]]
    new_idx = torch.cat([new_rows.unsqueeze(0), idx[1:, keep]], dim=0)

    new_size = (n_selected,) + tuple(v.shape[1:])
    return torch.sparse_coo_tensor(new_idx, vals[keep], new_size, device=v.device, dtype=v.dtype).coalesce()


def _device_of(x: Indexable) -> torch.device:
    if isinstance(x, Tensor):
        return x.device
    # DictBatch: take first tensor device
    for v in x.values():
        return v.device
    return torch.device("cpu")


@define(slots=True)
class Filter:
    """
    Filter description for FilteredTensor.

    - `shape` is the *virtual* shape of filtered dimensions.
    - `slices` maps virtual dims -> stored dims (with ints removing dims).
    - `mask` is optional and applies on dim0 *after slicing* (sliced space).
    """

    slices: Slices = field(converter=_normalize_slices)
    shape: tuple[int, ...] = field()
    mask: Tensor | None = field(default=None)

    def __attrs_post_init__(self) -> None:
        if self.mask is not None:
            if not isinstance(self.mask, Tensor) or self.mask.dtype is not torch.bool:
                raise TypeError("Filter.mask must be a torch.bool Tensor or None")
            if self.mask.ndim != 1:
                raise ValueError("Filter.mask must be 1D (doc/batch mask)")

    @property
    def virtual_shape(self) -> tuple[int, ...]:
        return tuple(self.shape)

    @property
    def filtered_ndim(self) -> int:
        return len(self.shape)

    def sliced_doc_len(self) -> int:
        if not self.shape:
            raise ValueError("Filter.shape must include at least the batch/doc dim")
        sl0 = self.slices[0] if self.slices else None
        return _slice_len(self.shape[0], sl0)

    def normalized_mask_in_sliced_space(self) -> Tensor | None:
        """
        If mask is global-length but slices[0] selects a chunk, slice it down.
        """
        if self.mask is None:
            return None
        if not self.shape:
            return self.mask

        doc_virtual = self.shape[0]
        sl0 = self.slices[0] if self.slices else None
        doc_sliced = _slice_len(doc_virtual, sl0)

        if self.mask.shape[0] == doc_sliced:
            return self.mask
        if self.mask.shape[0] == doc_virtual:
            if sl0 is None:
                return self.mask
            match sl0:
                case slice() as s:
                    return self.mask[s]
                case int(i):
                    return self.mask[i : i + 1]
                case None:
                    return self.mask
        raise ValueError(
            f"Mask length mismatch: mask={self.mask.shape[0]}, "
            f"virtual_doc={doc_virtual}, sliced_doc={doc_sliced}, slices[0]={sl0}"
        )

    def to(self, *args, **kwargs) -> Self:
        m = self.mask
        if m is not None:
            m = m.to(*args, **kwargs)
        return self.__class__(slices=self.slices, mask=m, shape=self.shape)

    def writeat(self, target: Indexable, value: Indexable, *, _value_kw: Any = None) -> None:
        """
        Write `value` into `target` at the positions selected by (slices, mask).

        Note: accepts the old keyword call style:
            filter.writeat(target=..., value=...)
        """
        # Preserve compatibility with calls like writeat(target=..., value=...)
        if _value_kw is not None:
            value = _value_kw  # type: ignore[assignment]

        idx = _as_index_tuple(self.slices)
        m = self.normalized_mask_in_sliced_space()

        if isinstance(target, DictBatch):
            if idx and len(idx) > 1:
                non_batch = [
                    s for s in idx[1:] if not (isinstance(s, slice) and _is_full_slice(s))
                ]
                if non_batch:
                    raise ValueError("Cannot writeat into DictBatch with non-batch slicing")
            sl0 = idx[0] if idx else slice(None)
            if m is None:
                target[sl0] = value  # type: ignore[index]
            else:
                # Target slice then mask.
                sub = target[sl0]
                sub[m] = value  # type: ignore[index]
                target[sl0] = sub  # type: ignore[index]
            return

        view = target[idx] if idx else target
        if m is None:
            view[...] = value  # type: ignore[index]
        else:
            view[m] = value  # type: ignore[index]


@define(slots=True)
class FilteredTensor:
    """
    Tensor/DictBatch with a virtual "larger" batch/doc space, represented via:

      virtual tensor (shape = Filter.shape + trailing dims)
        -> apply Filter.slices (usually chunk selection / feature index)
        -> apply Filter.mask (doc selection)  [optional]
        -> stored value (self.value)

    This matches how your code uses FilteredTensor:
      - chunk doc-range slices + optional per-chunk doc mask
      - feature tensors: fixed feature id via int slice in last dim
      - preserve trailing dims for sparse indices externalize/internalize
    """

    value: Indexable
    filter: Filter

    # ---- shape helpers ----

    @property
    def slices(self) -> Slices:
        return self.filter.slices

    @property
    def mask(self) -> Filter:
        return self.filter

    @property
    def filter_mask(self) -> Tensor | None:
        return self.filter.mask

    def _compute_filtered_shape_prefix(self):
        return self.virtual_shape

    @property
    def virtual_shape(self) -> tuple[int, ...]:
        return self.filter.virtual_shape

    @cached_property
    def _stored_ndim(self) -> int:
        """
        Number of *stored* dims corresponding to filtered dims (excluding int-sliced dims).
        Does not count trailing dims.
        """
        n = 0
        for i in range(len(self.filter.shape)):
            sl = self.slices[i] if i < len(self.slices) else None
            if isinstance(sl, int):
                continue
            n += 1
        return n

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Full conceptual shape = filter.shape + trailing dims from value.

        Trailing dims are those not part of filter.shape (e.g. seq_len, d_dict for chunk tensors).
        """
        if isinstance(self.value, DictBatch):
            # DictBatch has only batch dim shape semantics.
            trailing: tuple[int, ...] = ()
        else:
            trailing = tuple(self.value.shape[self._stored_ndim :])
        return tuple(self.filter.shape) + trailing

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.value, Tensor) and self.value.is_sparse

    @property
    def device(self) -> torch.device:
        return _device_of(self.value)

    @property
    def n_selected(self) -> int:
        """
        Number of selected docs in the *sliced* space.

        If mask exists: mask.sum()
        else: doc length after slice[0]
        """
        m = self.filter.normalized_mask_in_sliced_space()
        if m is not None:
            return int(m.sum().item())
        return self.filter.sliced_doc_len()

    # ---- mask index maps (used for index externalization) ----

    @cached_property
    def _mask_true_positions(self) -> Tensor | None:
        m = self.filter.normalized_mask_in_sliced_space()
        if m is None:
            return None
        return m.nonzero(as_tuple=False).flatten()

    @cached_property
    def _mask_inverse_map(self) -> Tensor | None:
        m = self.filter.normalized_mask_in_sliced_space()
        if m is None:
            return None
        n_selected = int(m.sum().item())
        inv = torch.full((m.numel(),), -1, dtype=torch.long, device=m.device)
        inv[m] = torch.arange(n_selected, device=m.device)
        return inv

    # ---- constructors ----

    @classmethod
    def from_value_and_mask(cls, value: Indexable, mask_obj: Tensor | None) -> Self:
        if mask_obj is None:
            # Trivial full mask.
            bs = value.shape[0] if isinstance(value, Tensor) else value.batch_size
            mask_obj = torch.ones(bs, dtype=torch.bool, device=_device_of(value))
        filt = Filter(slices=(), mask=mask_obj.to(dtype=torch.bool), shape=(mask_obj.shape[0],))
        # Here, mask is in sliced space already, and value is assumed already masked.
        return cls(value=value, filter=filt)

    @classmethod
    def from_unmasked_value(
        cls,
        value: Indexable,
        slicing: Sequence[slice | int | None] | None = None,
        mask: Tensor | None = None,
        filter_obj: Filter | NamedFilter | Tensor | None = None,
        presliced: bool = False,
    ) -> Self:
        """
        Build a FilteredTensor by applying slicing (optional) then mask (optional).

        - If `presliced=False`, slicing is applied to `value`.
        - Mask is always interpreted as applying on dim0 *after slicing*.
        """
        # Resolve filter_obj -> (slices, mask, shape)
        shape: tuple[int, ...] | None = None
        if filter_obj is not None:
            if isinstance(filter_obj, Filter):
                slicing = filter_obj.slices
                mask = filter_obj.mask
                shape = filter_obj.shape
            elif isinstance(filter_obj, NamedFilter):
                slicing = ()
                mask = filter_obj.filter
                shape = (int(mask.shape[0]),)
            elif isinstance(filter_obj, Tensor):
                slicing = ()
                mask = filter_obj
                shape = (int(mask.shape[0]),)
            else:
                raise TypeError(f"Unsupported filter_obj type: {type(filter_obj)}")

        slices = _normalize_slices(slicing)

        if shape is None:
            # Infer shape for filtered dims from the (un-sliced) input if possible.
            if presliced:
                raise ValueError("presliced=True requires filter_obj with explicit shape")
            if isinstance(value, DictBatch):
                base = (value.batch_size,)
            else:
                base = tuple(value.shape)
            if mask is not None:
                shape = (int(mask.shape[0]),)
            else:
                shape = base[: len(slices)] if slices else (base[0],)

        filt = Filter(slices=slices, mask=mask, shape=shape)

        # Apply slicing if needed.
        if slices and not presliced:
            value = _apply_slices(value, slices)

        # Normalize mask into sliced space, then apply it.
        m = filt.normalized_mask_in_sliced_space()
        if m is not None:
            m = m.to(device=_device_of(value), dtype=torch.bool)
            if isinstance(value, DictBatch):
                value = value[m]
            else:
                if value.is_sparse:
                    value = _sparse_row_mask(value, m)
                else:
                    value = value[m]

            filt = Filter(slices=slices, mask=m, shape=shape)

        return cls(value=value, filter=filt)

    # ---- index mapping ----

    def internalize_indices(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        """
        Map full-space indices -> stored-space indices.

        Indices are shaped (ndim, n_points) and may include trailing dims beyond Filter.shape.
        """
        if indices.ndim != 2:
            raise ValueError("indices must be a 2D tensor shaped (ndim, n_points)")
        if indices.dtype is not torch.long:
            indices = indices.to(dtype=torch.long)

        filt_ndim = len(self.filter.shape)
        in_ndim = indices.shape[0]
        n = indices.shape[1]
        device = indices.device

        idx_filtered = indices[: min(filt_ndim, in_ndim)]
        trailing = indices[filt_ndim:] if in_ndim > filt_ndim else None

        valid = torch.ones(n, dtype=torch.bool, device=device)
        inner_rows: list[Tensor] = []

        # For dims we have, apply slicing mapping and validate int dims if provided.
        for dim_i in range(idx_filtered.shape[0]):
            spec = self.slices[dim_i] if dim_i < len(self.slices) else None
            coord = idx_filtered[dim_i]

            match spec:
                case int(val):
                    valid &= coord == val
                case slice() as s:
                    start, stop, step = _normalize_slice(s, self.filter.shape[dim_i])
                    valid &= (coord >= start) & (coord < stop)
                    valid &= ((coord - start) % step) == 0
                    inner_rows.append((coord - start) // step)
                case None:
                    inner_rows.append(coord)
                case _:
                    raise TypeError(f"Unexpected slice spec: {type(spec)}")

        inner = (
            torch.stack(inner_rows, dim=0)
            if inner_rows
            else torch.empty((0, n), dtype=torch.long, device=device)
        )

        # Apply doc-mask mapping on first stored dim (if present).
        m = self.filter.normalized_mask_in_sliced_space()
        if m is not None and inner.shape[0] >= 1:
            m = m.to(device=device)
            inv = self._mask_inverse_map
            if inv is None:
                raise RuntimeError("mask inverse map missing")
            inv = inv.to(device=device)

            doc = inner[0]
            valid &= (doc >= 0) & (doc < m.shape[0])
            doc_clamped = doc.clamp(0, m.shape[0] - 1)
            valid &= m[doc_clamped]

            mapped = inv[doc_clamped]
            inner = inner.clone()
            inner[0] = mapped

        if trailing is not None and trailing.numel() > 0:
            inner = torch.cat([inner, trailing], dim=0)

        return inner, valid

    def externalize_indices(self, indices: Tensor) -> Tensor:
        """
        Map stored-space indices -> full-space indices.

        Input indices may include trailing dims beyond stored filtered dims; they are preserved.
        """
        if indices.ndim != 2:
            raise ValueError("indices must be a 2D tensor shaped (ndim, n_points)")
        if indices.dtype is not torch.long:
            indices = indices.to(dtype=torch.long)

        filt_ndim = len(self.filter.shape)
        stored_ndim = self._stored_ndim
        in_ndim = indices.shape[0]

        idx_filtered = indices[: min(stored_ndim, in_ndim)]
        trailing = indices[stored_ndim:] if in_ndim > stored_ndim else None

        # Unmask doc dimension.
        m = self.filter.normalized_mask_in_sliced_space()
        if m is not None and idx_filtered.shape[0] >= 1:
            tp = self._mask_true_positions
            if tp is None:
                raise RuntimeError("mask true positions missing")
            tp = tp.to(device=indices.device)
            doc = idx_filtered[0]
            idx_filtered = idx_filtered.clone()
            idx_filtered[0] = tp[doc]

        # Rebuild full filtered dims, inserting int dims.
        out_rows: list[Tensor] = []
        p = 0
        for dim_i in range(filt_ndim):
            spec = self.slices[dim_i] if dim_i < len(self.slices) else None
            match spec:
                case int(val):
                    out_rows.append(
                        torch.full(
                            (idx_filtered.shape[1],),
                            int(val),
                            device=indices.device,
                            dtype=torch.long,
                        )
                    )
                case slice() as s:
                    inner = idx_filtered[p]
                    p += 1
                    start, _stop, step = _normalize_slice(s, self.filter.shape[dim_i])
                    out_rows.append(inner * step + start)
                case None:
                    inner = idx_filtered[p]
                    p += 1
                    out_rows.append(inner)
                case _:
                    raise TypeError(f"Unexpected slice spec: {type(spec)}")

        out = torch.stack(out_rows, dim=0) if out_rows else idx_filtered
        if trailing is not None and trailing.numel() > 0:
            out = torch.cat([out, trailing], dim=0)
        return out

    # ---- indexing / selection helpers ----

    def index_where_valid(self, indices: Tensor) -> tuple[Indexable, Tensor]:
        inner, valid = self.internalize_indices(indices)
        inner_valid = inner[:, valid]
        if isinstance(self.value, DictBatch):
            return self.value[inner_valid[0]], valid

        if self.value.is_sparse:
            # We only support row selection for sparse values here.
            return self.value.index_select(0, inner_valid[0]).coalesce(), valid

        return self.value[tuple(inner_valid)], valid

    def index_select(self, index: Tensor, dim: int = 0) -> Indexable:
        if dim != 0:
            raise NotImplementedError("FilteredTensor.index_select only supports dim=0")

        if index.ndim != 1:
            raise ValueError("index must be 1D")

        inner, valid = self.internalize_indices(index.unsqueeze(0))
        if not bool(valid.all().item()):
            raise IndexError("Some indices are outside the filtered region")

        inner_idx = inner[0]
        if isinstance(self.value, DictBatch):
            return self.value[inner_idx]
        if self.value.is_sparse:
            return self.value.index_select(0, inner_idx).coalesce()
        return self.value.index_select(0, inner_idx)

    # ---- mask composition ----

    @overload
    def mask_by_other(
        self,
        other: Tensor | Self,
        *,
        return_ft: Literal[True],
        presliced: bool = False,
        value_like: bool = False,
    ) -> Self: ...

    @overload
    def mask_by_other(
        self,
        other: Tensor | Self,
        *,
        return_ft: Literal[False] = False,
        presliced: bool = False,
        value_like: bool = False,
    ) -> Indexable: ...

    def mask_by_other(
        self,
        other: Tensor | Filter | Self,
        *,
        return_ft: bool = False,
        presliced: bool = False,
        value_like: bool = False,
    ) -> Indexable | Self:
        """
        Apply an additional doc-mask.

        Args:
          other:
            - Tensor[bool]
            - Filter
            - FilteredTensor (we use its filter.mask)
          presliced:
            If True, `other` is defined in sliced space (after our slices[0]).
            If False, `other` is defined in virtual space (length == filter.shape[0]).
          value_like:
            If True, `other` is defined in *value* space (length == self.value.shape[0]).
        """
        if isinstance(other, FilteredTensor):
            if other.filter.mask is None:
                raise ValueError("Cannot use a FilteredTensor with no mask as a mask source")
            other_mask = other.filter.mask
        elif isinstance(other, Filter):
            if other.mask is None:
                return self if return_ft else self.value
            other_mask = other.mask
        else:
            other_mask = other

        if not isinstance(other_mask, Tensor) or other_mask.dtype is not torch.bool:
            other_mask = other_mask.to(dtype=torch.bool)

        # Move to this device for actual masking.
        other_mask = other_mask.to(device=self.device)

        current_mask_sliced = self.filter.normalized_mask_in_sliced_space()
        doc_virtual_len = self.filter.shape[0]
        sl0 = self.slices[0] if self.slices else None
        doc_sliced_len = _slice_len(doc_virtual_len, sl0)

        if value_like:
            # other_mask is in value space: length == self.value.shape[0]
            other_value = other_mask
            if isinstance(self.value, DictBatch):
                new_value = self.value[other_value]
            else:
                new_value = (
                    _sparse_row_mask(self.value, other_value)
                    if self.value.is_sparse
                    else self.value[other_value]
                )

            if current_mask_sliced is None:
                # value space == sliced space
                new_mask_sliced = other_value
            else:
                # Compose: new_mask_sliced = current_mask_sliced with subset applied inside True positions.
                true_pos = current_mask_sliced.nonzero(as_tuple=False).flatten()
                if true_pos.numel() != other_value.numel():
                    raise ValueError(
                        "value_like mask must match number of currently-selected docs"
                    )
                new_mask_sliced = current_mask_sliced.clone()
                new_mask_sliced[true_pos] = other_value

            new_filter = Filter(slices=self.slices, mask=new_mask_sliced, shape=self.filter.shape)
            return self.__class__(value=new_value, filter=new_filter) if return_ft else new_value

        # Not value_like: compute other mask in sliced space.
        if presliced:
            other_sliced = other_mask
        else:
            # other is in virtual space (global docs). Slice it if needed.
            if other_mask.shape[0] == doc_virtual_len:
                if sl0 is None:
                    other_sliced = other_mask
                elif isinstance(sl0, slice):
                    other_sliced = other_mask[sl0]
                elif isinstance(sl0, int):
                    other_sliced = other_mask[sl0 : sl0 + 1]
                else:
                    other_sliced = other_mask
            else:
                other_sliced = other_mask

        if other_sliced.shape[0] != doc_sliced_len:
            raise ValueError(
                f"Mask length mismatch in sliced space: got {other_sliced.shape[0]}, expected {doc_sliced_len}"
            )

        if current_mask_sliced is None:
            # No current mask: apply other_sliced directly.
            other_value = other_sliced
            new_mask_sliced = other_sliced
        else:
            other_value = other_sliced[current_mask_sliced]
            new_mask_sliced = current_mask_sliced & other_sliced

        if isinstance(self.value, DictBatch):
            new_value = self.value[other_value]
        else:
            new_value = (
                _sparse_row_mask(self.value, other_value)
                if self.value.is_sparse
                else self.value[other_value]
            )

        new_filter = Filter(slices=self.slices, mask=new_mask_sliced, shape=self.filter.shape)
        return self.__class__(value=new_value, filter=new_filter) if return_ft else new_value

    def filter_inactive_docs(self) -> Self:
        """
        Refine to only docs where inner value has any nonzero.

        (Used heavily by patching code.)
        """
        if isinstance(self.value, DictBatch):
            raise TypeError("filter_inactive_docs not supported for DictBatch")

        if self.value.is_sparse:
            v = self.value.coalesce()
            active = torch.zeros(v.shape[0], dtype=torch.bool, device=v.device)
            active[v.indices()[0].unique()] = True
        else:
            active = self.value != 0
            while active.ndim > 1:
                active = active.any(dim=-1)

        return self.mask_by_other(active, return_ft=True, value_like=True)

    # ---- wrapping helpers ----

    def to_filtered_like_self(
        self,
        t: Indexable,
        *,
        presliced: bool = True,
        premasked: bool = True,
        ndim: int | None = None,
    ) -> Self:
        """
        Wrap a new tensor/DictBatch `t` with the same filter semantics.

        `presliced/premasked` describe whether `t` is already in stored space.
        """
        filt = self.filter
        if ndim is not None:
            # Truncate filtered dims.
            new_shape = filt.shape[:ndim]
            new_slices = self.slices[:ndim]
            new_mask = filt.mask if (ndim >= 1) else None
            filt = Filter(slices=new_slices, mask=new_mask, shape=new_shape)

        if not presliced and filt.slices:
            t = _apply_slices(t, filt.slices)

        if not premasked and filt.normalized_mask_in_sliced_space() is not None:
            m = filt.normalized_mask_in_sliced_space().to(device=_device_of(t))
            if isinstance(t, DictBatch):
                t = t[m]
            else:
                t = _sparse_row_mask(t, m) if t.is_sparse else t[m]

        return self.__class__(value=t, filter=filt)

    def apply_to_inner(self, func: Callable[[Tensor], Tensor], cut_to_ndim: int | None = None) -> Self:
        if isinstance(self.value, DictBatch):
            new_value = self.value.apply_func(func)
        else:
            new_value = func(self.value)
        return self.to_filtered_like_self(new_value, presliced=True, premasked=True, ndim=cut_to_ndim)

    def clone(self) -> Self:
        return self.__class__(value=self.value.clone(), filter=self.filter)

    def __mul__(self, other: Any) -> Self:
        if isinstance(other, FilteredTensor):
            other = other.value
        return self.apply_to_inner(lambda x: x * other)

    def __rmul__(self, other: Any) -> Self:
        return self.__mul__(other)

    def __imul__(self, other: Any) -> Self:
        if isinstance(other, FilteredTensor):
            other = other.value
        self.value *= other
        return self

    def __add__(self, other: Any) -> Self:
        if isinstance(other, FilteredTensor):
            other = other.value
        return self.apply_to_inner(lambda x: x + other)

    def __radd__(self, other: Any) -> Self:
        return self.__add__(other)

    def __iadd__(self, other: Any) -> Self:
        if isinstance(other, FilteredTensor):
            other = other.value
        self.value += other
        return self

    # ---- materialization ----

    def to_dense_unfiltered(self, fill_value: float = 0.0) -> Tensor:
        """
        Materialize full virtual tensor (dense), filling unselected positions.
        """
        if isinstance(self.value, DictBatch):
            raise TypeError("to_dense_unfiltered not supported for DictBatch")

        inner = self.value.to_dense() if self.value.is_sparse else self.value
        out = torch.full(self.shape, fill_value, dtype=inner.dtype, device=inner.device)
        self.filter.writeat(out, inner)
        return out

    def to_sparse_unfiltered(self) -> Tensor:
        if isinstance(self.value, DictBatch):
            raise TypeError("to_sparse_unfiltered not supported for DictBatch")
        return torch.sparse_coo_tensor(
            indices=self.indices(),
            values=self.values(),
            size=self.shape,
            dtype=self.value.dtype,
            device=self.value.device,
        ).coalesce()

    def to_sparse(self) -> Self:
        if isinstance(self.value, DictBatch):
            raise TypeError("to_sparse is only defined for tensor values")
        if self.is_sparse:
            return self
        return self.to_filtered_like_self(self.value.to_sparse_coo(), presliced=True, premasked=True)

    def to_dense(self) -> Self:
        if isinstance(self.value, DictBatch):
            return self
        if not self.value.is_sparse:
            return self
        return self.to_filtered_like_self(self.value.to_dense(), presliced=True, premasked=True)

    def writeat(self, target: Indexable, value: Indexable | None = None) -> None:
        """
        Write into target using our filter.
        """
        if value is None:
            value = self.value
        self.filter.writeat(target, value)

    # ---- sparse helpers ----

    def indices(self) -> Tensor:
        if not isinstance(self.value, Tensor) or not self.value.is_sparse:
            raise TypeError("indices() only valid when value is a sparse tensor")
        inner = self.value.coalesce().indices()
        return self.externalize_indices(inner)

    def values(self) -> Tensor:
        if not isinstance(self.value, Tensor) or not self.value.is_sparse:
            raise TypeError("values() only valid when value is a sparse tensor")
        return self.value.coalesce().values()

    def nonzero(self) -> Tensor:
        if isinstance(self.value, DictBatch):
            raise TypeError("nonzero() not supported for DictBatch")
        if self.value.is_sparse:
            return self.indices().T
        inner = self.value.nonzero().T
        return self.externalize_indices(inner).T

    # ---- device helpers ----

    def to(self, *args, **kwargs) -> Self:
        if isinstance(self.value, DictBatch):
            new_value = self.value.to(*args, **kwargs)
        else:
            new_value = self.value.to(*args, **kwargs)

        m = self.filter.mask
        if m is not None:
            # Keep boolean dtype.
            m = m.to(*args, dtype=torch.bool, **{k: v for k, v in kwargs.items() if k != "dtype"})

        new_filter = Filter(slices=self.filter.slices, mask=m, shape=self.filter.shape)
        return self.__class__(value=new_value, filter=new_filter)

    def cuda(self) -> Self:
        return self.to(device="cuda")

    def __getitem__(self, key: Tensor):
        if not isinstance(key, Tensor) or key.dtype != torch.long or key.ndim != 2:
            raise NotImplementedError("Only 2D long index tensors are supported")
        inner, valid = self.internalize_indices(key)
        if not bool(valid.all().item()):
            raise IndexError("Some indices are outside the filtered region")
        if isinstance(self.value, DictBatch):
            # Only batch selection is meaningful here.
            return self.value[inner[0]]
        return self.value[inner.unbind()]