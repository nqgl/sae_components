from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, Self, overload

import torch
from attrs import define, field
from torch import Tensor

from saeco.data.dict_batch.dict_batch import DictBatch

type Wrappable = Tensor | DictBatch | MaskedTensor | SlicedTensor
type Indexable = Tensor | DictBatch


def slice_shape(
    input_shape: tuple[int, ...], slices: Sequence[slice | int | None]
) -> tuple[int, ...]:
    """Compute output shape after applying slices."""
    slices = tuple(slices) + (None,) * (len(input_shape) - len(slices))
    out: list[int] = []
    for slc, dim in zip(slices, input_shape, strict=False):
        match slc:
            case int():
                continue
            case None:
                out.append(dim)
            case slice(start, stop, step):
                start = start or 0
                stop = min(stop, dim) if stop is not None else dim
                step = step or 1
                out.append((stop - start + step - 1) // step)
    return tuple(out)


# ============================================================================
# FilterBase ABC
# ============================================================================


class FilterBase(ABC):
    """
    Abstract base for filter operations (Mask, Slicing, etc.)

    A filter describes a transformation from a virtual tensor space
    to a smaller inner space, with bidirectional index mapping.
    """

    @property
    @abstractmethod
    def virtual_shape(self) -> tuple[int, ...]:
        """Shape of the conceptual full tensor this filter operates on."""
        ...

    @property
    @abstractmethod
    def inner_shape(self) -> tuple[int, ...]:
        """Shape after applying this filter."""
        ...

    @abstractmethod
    def apply(self, tensor: Indexable) -> Indexable:
        """Apply this filter to a tensor/DictBatch of virtual_shape."""
        ...

    @abstractmethod
    def writeat(self, target: Indexable, value: Indexable) -> None:
        """
        Write `value` into `target` at the positions selected by this filter.

        Args:
            target: Tensor/DictBatch of virtual_shape to write into
            value: Tensor/DictBatch of inner_shape containing values to write
        """
        ...

    @abstractmethod
    def internalize_indices(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        """
        Convert virtual-space indices to inner-space indices.

        Args:
            indices: (ndim, n_points) coordinates in virtual space

        Returns:
            (inner_indices, valid_mask) where valid_mask indicates
            which points survived the filter
        """
        ...

    @abstractmethod
    def externalize_indices(self, indices: Tensor) -> Tensor:
        """
        Convert inner-space indices back to virtual-space indices.

        Args:
            indices: (ndim, n_points) coordinates in inner space

        Returns:
            Coordinates in virtual space
        """
        ...

    @abstractmethod
    def to(self, *args, **kwargs) -> Self:
        """Move any internal tensors to specified device/dtype."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.virtual_shape} -> {self.inner_shape})"


# ============================================================================
# Slicing
# ============================================================================


@define
class Slicing(FilterBase):
    """
    Describes a slicing operation on a tensor of known shape.

    For DictBatch, slicing is applied to all fields uniformly.
    Only batch-dimension (dim 0) slicing is well-defined for DictBatch.
    """

    slices: tuple[slice | int | None, ...] = field(converter=tuple)
    _virtual_shape: tuple[int, ...] = field(alias="virtual_shape")

    @property
    def virtual_shape(self) -> tuple[int, ...]:
        return self._virtual_shape

    @property
    def inner_shape(self) -> tuple[int, ...]:
        return slice_shape(self._virtual_shape, self.slices)

    @property
    def ndim_removed(self) -> int:
        """Number of dimensions removed by integer indexing."""
        return sum(isinstance(s, int) for s in self.slices)

    @property
    def normalized(self) -> tuple[slice | int, ...]:
        """Slicing tuple with None -> slice(None)."""
        return tuple(sl if sl is not None else slice(None) for sl in self.slices)

    @property
    def batch_slice(self) -> slice | int | None:
        """The slice applied to dimension 0 (batch dimension)."""
        return self.slices[0] if self.slices else None

    def apply(self, tensor: Indexable) -> Indexable:
        match tensor:
            case DictBatch():
                return self._apply_dictbatch(tensor)
            case Tensor():
                return tensor[self.normalized]

    def _apply_dictbatch(self, db: DictBatch) -> DictBatch:
        """Apply slicing to DictBatch - only batch dim slicing supported."""
        if len(self.slices) > 1:
            non_batch = [
                s for s in self.slices[1:] if s is not None and s != slice(None)
            ]
            if non_batch:
                raise ValueError(
                    f"DictBatch only supports batch-dimension slicing, "
                    f"got non-trivial slices beyond dim 0: {self.slices[1:]}"
                )
        batch_sl = self.batch_slice
        if batch_sl is None:
            batch_sl = slice(None)
        return db[batch_sl]

    def writeat(self, target: Indexable, value: Indexable) -> None:
        match target:
            case DictBatch():
                self._writeat_dictbatch(target, value)
            case Tensor():
                target[self.normalized] = value

    def _writeat_dictbatch(self, target: DictBatch, value: DictBatch) -> None:
        """Write into DictBatch at sliced positions."""
        if not isinstance(value, DictBatch):
            raise TypeError(f"Expected DictBatch value, got {type(value)}")

        batch_sl = self.batch_slice
        if batch_sl is None:
            batch_sl = slice(None)

        # DictBatch should support item assignment or we iterate fields
        if hasattr(target, "__setitem__"):
            target[batch_sl] = value
        else:
            # Fallback: iterate over matching keys
            for key in value.keys():
                if key in target:
                    target[key][batch_sl] = value[key]

    def internalize_indices(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        assert indices.ndim == 2
        device = indices.device

        starts: list[int] = []
        steps: list[int] = []
        stops: list[int] = []
        int_dims: dict[int, int] = {}

        for i, sl in enumerate(self.slices):
            match sl:
                case int(val):
                    int_dims[i] = val
                case None:
                    starts.append(0)
                    steps.append(1)
                    stops.append(self._virtual_shape[i])
                case slice(start, stop, step):
                    starts.append(start or 0)
                    steps.append(step or 1)
                    stops.append(stop if stop is not None else self._virtual_shape[i])

        valid = torch.ones(indices.shape[1], dtype=torch.bool, device=device)

        # Check integer-indexed dimensions
        for dim_idx, expected in int_dims.items():
            if dim_idx < indices.shape[0]:
                valid &= indices[dim_idx] == expected

        # Remove int-indexed dims
        keep_dims = [i for i in range(indices.shape[0]) if i not in int_dims]
        inner_indices = indices[keep_dims] if keep_dims else indices

        if starts:
            starts_t = torch.tensor(starts, device=device)
            steps_t = torch.tensor(steps, device=device)
            stops_t = torch.tensor(stops, device=device)
            n = min(len(starts), inner_indices.shape[0])

            valid &= (inner_indices[:n] >= starts_t[:n, None]).all(dim=0)
            valid &= (inner_indices[:n] < stops_t[:n, None]).all(dim=0)
            valid &= (
                (inner_indices[:n] - starts_t[:n, None]) % steps_t[:n, None] == 0
            ).all(dim=0)

            inner_indices = inner_indices.clone()
            inner_indices[:n] = (inner_indices[:n] - starts_t[:n, None]) // steps_t[
                :n, None
            ]

        return inner_indices, valid

    def externalize_indices(self, indices: Tensor) -> Tensor:
        assert indices.ndim == 2
        device = indices.device

        starts: list[int] = []
        steps: list[int] = []
        int_dims: list[tuple[int, int]] = []

        for i, sl in enumerate(self.slices):
            match sl:
                case int(val):
                    int_dims.append((i, val))
                case None:
                    starts.append(0)
                    steps.append(1)
                case slice(start, _, step):
                    starts.append(start or 0)
                    steps.append(step or 1)

        outer = indices.clone()
        if starts:
            starts_t = torch.tensor(starts, device=device)
            steps_t = torch.tensor(steps, device=device)
            n = min(len(starts), indices.shape[0])
            outer[:n] = indices[:n] * steps_t[:n, None] + starts_t[:n, None]

        # Reinsert integer dimensions
        for dim_idx, val in sorted(int_dims):
            outer = torch.cat(
                [
                    outer[:dim_idx],
                    torch.full(
                        (1, outer.shape[1]), val, device=device, dtype=torch.long
                    ),
                    outer[dim_idx:],
                ]
            )

        return outer

    def to(self, *args, **kwargs) -> Self:
        return self  # Slicing has no tensors


# ============================================================================
# Mask
# ============================================================================


@define
class Mask(FilterBase):
    """
    Describes a boolean mask operation.

    For DictBatch, only 1D masks over the batch dimension are supported.
    Caches derived quantities for efficient repeated index operations.
    """

    mask: Tensor = field()
    _n_selected: int = field(init=False)
    _true_positions: Tensor = field(init=False, repr=False)
    _inverse_map: Tensor = field(init=False, repr=False)

    @_n_selected.default
    def _compute_n_selected(self) -> int:
        return int(self.mask.sum().item())

    @_true_positions.default
    def _compute_true_positions(self) -> Tensor:
        return self.mask.nonzero()  # (n_selected, mask_ndim)

    @_inverse_map.default
    def _compute_inverse_map(self) -> Tensor:
        flat = self.mask.flatten()
        inv = torch.full((flat.numel(),), -1, dtype=torch.long, device=self.mask.device)
        inv[flat] = torch.arange(self._n_selected, device=self.mask.device)
        return inv

    @property
    def virtual_shape(self) -> tuple[int, ...]:
        return tuple(self.mask.shape)

    @property
    def inner_shape(self) -> tuple[int, ...]:
        return (self._n_selected,)

    @property
    def n_selected(self) -> int:
        return self._n_selected

    @property
    def device(self) -> torch.device:
        return self.mask.device

    @property
    def is_batch_mask(self) -> bool:
        """True if this is a 1D mask suitable for DictBatch."""
        return self.mask.ndim == 1

    def apply(self, tensor: Indexable) -> Indexable:
        match tensor:
            case DictBatch():
                return self._apply_dictbatch(tensor)
            case Tensor() if tensor.is_sparse:
                return _sparse_mask(tensor.coalesce(), self.mask)
            case Tensor():
                return tensor[self.mask]

    def _apply_dictbatch(self, db: DictBatch) -> DictBatch:
        """Apply mask to DictBatch - only 1D batch masks supported."""
        if not self.is_batch_mask:
            raise ValueError(
                f"DictBatch requires 1D mask over batch dimension, "
                f"got {self.mask.ndim}D mask of shape {self.mask.shape}"
            )
        if len(db) != self.mask.shape[0]:
            raise ValueError(
                f"Mask length {self.mask.shape[0]} doesn't match "
                f"DictBatch batch size {len(db)}"
            )
        return db[self.mask]

    def writeat(self, target: Indexable, value: Indexable) -> None:
        match target:
            case DictBatch():
                self._writeat_dictbatch(target, value)
            case Tensor() if target.is_sparse:
                raise NotImplementedError("writeat not supported for sparse targets")
            case Tensor():
                target[self.mask] = value

    def _writeat_dictbatch(self, target: DictBatch, value: DictBatch) -> None:
        """Write into DictBatch at masked positions."""
        if not isinstance(value, DictBatch):
            raise TypeError(f"Expected DictBatch value, got {type(value)}")
        if not self.is_batch_mask:
            raise ValueError("DictBatch requires 1D mask over batch dimension")

        # Write each field
        if hasattr(target, "__setitem__") and hasattr(target, "keys"):
            # If DictBatch supports direct masked assignment
            try:
                target[self.mask] = value
                return
            except (TypeError, NotImplementedError):
                pass

        # Fallback: iterate fields
        for key in value.keys():
            if key in target:
                target[key][self.mask] = value[key]

    def internalize_indices(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        assert indices.ndim == 2
        device = indices.device
        mask_ndim = self.mask.ndim

        # Bounds check
        shape_t = torch.tensor(self.mask.shape, device=device)
        mask_indices = indices[:mask_ndim]

        valid = (mask_indices >= 0).all(dim=0)
        valid &= (mask_indices < shape_t[:, None]).all(dim=0)

        # Compute flat index
        strides = self._strides(device)
        flat_idx = (mask_indices * strides[:, None]).sum(dim=0)
        flat_idx_clamped = flat_idx.clamp(0, self.mask.numel() - 1)

        # Check mask is True there
        valid &= self.mask.flatten()[flat_idx_clamped]

        # Map to inner index
        inner_first = self._inverse_map.to(device)[flat_idx_clamped]
        inner_indices = torch.cat([inner_first[None], indices[mask_ndim:]])

        return inner_indices, valid

    def externalize_indices(self, indices: Tensor) -> Tensor:
        assert indices.ndim == 2
        first_dim = indices[0]
        outer_prefix = self._true_positions.to(indices.device)[first_dim].T
        return torch.cat([outer_prefix, indices[1:]])

    def _strides(self, device: torch.device | str) -> Tensor:
        shape = self.mask.shape
        strides = []
        for i in range(len(shape)):
            strides.append(self.mask[i:].numel() // shape[i] if i < len(shape) else 1)
        return torch.tensor(strides, device=device, dtype=torch.long)

    def to(self, *args, **kwargs) -> Mask:
        new_mask = self.mask.to(*args, **kwargs)
        if new_mask is self.mask:
            return self
        return Mask(new_mask)

    # ---- Boolean operations ----

    def __and__(self, other: Mask) -> Mask:
        if self.mask.shape != other.mask.shape:
            raise ValueError(f"Shape mismatch: {self.mask.shape} vs {other.mask.shape}")
        return Mask(self.mask & other.mask)

    def __or__(self, other: Mask) -> Mask:
        if self.mask.shape != other.mask.shape:
            raise ValueError(f"Shape mismatch: {self.mask.shape} vs {other.mask.shape}")
        return Mask(self.mask | other.mask)

    def __invert__(self) -> Mask:
        return Mask(~self.mask)

    def __xor__(self, other: Mask) -> Mask:
        if self.mask.shape != other.mask.shape:
            raise ValueError(f"Shape mismatch: {self.mask.shape} vs {other.mask.shape}")
        return Mask(self.mask ^ other.mask)

    # ---- Refinement / composition ----

    def compose_inner(self, inner_mask: Mask) -> Mask:
        """
        Compose with a mask in the inner (already-masked) space.

        If self selects N elements, inner_mask must have shape (N,) or
        compatible, and the result selects the subset of self's selection.
        """
        if inner_mask.mask.shape[0] != self._n_selected:
            raise ValueError(
                f"Inner mask size {inner_mask.mask.shape[0]} doesn't match "
                f"current selection count {self._n_selected}"
            )

        # Scatter inner_mask back to virtual space
        new_flat = torch.zeros(
            self.mask.numel(),
            dtype=torch.bool,
            device=self.mask.device,
        )
        current_true_flat = self.mask.flatten().nonzero().squeeze(-1)
        new_flat[current_true_flat[inner_mask.mask]] = True

        return Mask(new_flat.view(self.mask.shape))


def _sparse_mask(v: Tensor, mask: Tensor) -> Tensor:
    """Apply boolean mask to sparse COO tensor."""
    indices = v.indices()
    values = v.values()
    mask_ndim = mask.ndim
    device = v.device

    strides = torch.tensor(
        [mask[i:].numel() // mask.shape[i] for i in range(mask_ndim)],
        device=device,
    )
    flat_idx = (indices[:mask_ndim] * strides[:, None]).sum(dim=0)
    keep = mask.flatten()[flat_idx]

    new_first = torch.zeros(mask.numel(), dtype=torch.long, device=device)
    new_first[mask.flatten()] = torch.arange(mask.sum().item(), device=device)

    new_indices = torch.cat(
        [
            new_first[flat_idx[keep]][None],
            indices[mask_ndim:, keep],
        ]
    )

    return torch.sparse_coo_tensor(
        new_indices,
        values[keep],
        (int(mask.sum().item()),) + v.shape[mask_ndim:],
    )


def _sparse_writeat(target: Tensor, mask: Tensor, value: Tensor) -> None:
    """
    Write values into sparse tensor at masked positions.

    This is tricky for sparse tensors - we need to update existing
    entries and potentially add new ones.
    """
    raise NotImplementedError(
        "writeat for sparse tensors requires careful handling of "
        "index merging - not yet implemented"
    )


# ============================================================================
# FilteredTensorBase ABC
# ============================================================================


class FilteredTensorBase[F: FilterBase](ABC):
    """
    Abstract base for tensors with a virtual larger shape.

    Parameterized by the filter type F (Mask, Slicing, etc.)
    """

    @property
    @abstractmethod
    def inner_value(self) -> Wrappable: ...

    @property
    @abstractmethod
    def filter(self) -> F: ...

    @property
    def virtual_shape(self) -> tuple[int, ...]:
        return self.filter.virtual_shape

    @property
    def inner_shape(self) -> tuple[int, ...]:
        return self.filter.inner_shape

    @property
    def root_value(self) -> Indexable:
        """Recursively unwrap to innermost tensor/DictBatch."""
        v = self.inner_value
        while isinstance(v, FilteredTensorBase):
            v = v.inner_value
        return v

    @property
    def is_dictbatch(self) -> bool:
        """True if the root value is a DictBatch."""
        return isinstance(self.root_value, DictBatch)

    def internalize_indices(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        return self.filter.internalize_indices(indices)

    def externalize_indices(self, indices: Tensor) -> Tensor:
        return self.filter.externalize_indices(indices)

    @abstractmethod
    def to(self, *args, **kwargs) -> Self: ...

    def full_internalize(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        """Recursively internalize through all nesting."""
        inner_idx, valid = self.internalize_indices(indices)
        if isinstance(self.inner_value, FilteredTensorBase):
            deeper_idx, deeper_valid = self.inner_value.full_internalize(inner_idx)
            combined_valid = valid.clone()
            combined_valid[valid] &= deeper_valid
            return deeper_idx, combined_valid
        return inner_idx, valid

    def full_externalize(self, indices: Tensor) -> Tensor:
        """Recursively externalize through all nesting."""
        if isinstance(self.inner_value, FilteredTensorBase):
            indices = self.inner_value.full_externalize(indices)
        return self.externalize_indices(indices)

    def writeat(self, target: Indexable, value: Indexable | None = None) -> None:
        """
        Write inner value (or provided value) into target at filtered positions.

        Args:
            target: Tensor/DictBatch of virtual_shape to write into
            value: Optional value to write; if None, uses self.inner_value
        """
        if value is None:
            value = _as_indexable(self.inner_value)
        self.filter.writeat(target, value)

    def full_writeat(self, target: Indexable, value: Indexable | None = None) -> None:
        """
        Recursively write through all nesting levels.

        For nested filters, this writes `value` through the innermost
        filter first, building up intermediate tensors.
        """
        if value is None:
            value = _as_indexable(self.inner_value)

        if isinstance(self.inner_value, FilteredTensorBase):
            # Need to build intermediate target at this level's inner shape
            intermediate = _make_empty_like(target, self.filter.inner_shape)
            self.inner_value.full_writeat(intermediate, value)
            self.filter.writeat(target, intermediate)
        else:
            self.filter.writeat(target, value)

    def to_dense(self, fill_value: float = 0.0) -> Indexable:
        """
        Materialize the full virtual tensor with fill_value in unselected positions.

        Only valid for Tensor inner values (not DictBatch).
        """
        root = self.root_value
        if isinstance(root, DictBatch):
            raise TypeError("to_dense not supported for DictBatch")

        result = torch.full(
            self.virtual_shape + root.shape[1:],
            fill_value,
            dtype=root.dtype,
            device=root.device,
        )
        self.full_writeat(result, _as_indexable(self.inner_value))
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}(filter={self.filter}, inner={type(self.inner_value).__name__})"


# ============================================================================
# SlicedTensor
# ============================================================================


@define
class SlicedTensor(FilteredTensorBase[Slicing]):
    """A tensor/DictBatch viewed through a Slicing filter."""

    _inner_value: Wrappable = field()
    _filter: Slicing = field(alias="filter")

    @property
    def inner_value(self) -> Wrappable:
        return self._inner_value

    @property
    def filter(self) -> Slicing:
        return self._filter

    # Convenient accessors
    @property
    def slicing(self) -> Slicing:
        return self._filter

    @property
    def slices(self) -> tuple[slice | int | None, ...]:
        return self._filter.slices

    def __attrs_post_init__(self):
        actual = _get_shape(self._inner_value)
        expected = self._filter.inner_shape
        if actual[: len(expected)] != expected:
            raise ValueError(
                f"Inner shape {actual} doesn't match slicing result {expected}"
            )

    @classmethod
    def from_unsliced(
        cls,
        value: Wrappable,
        slicing: Slicing | tuple[slice | int | None, ...],
        virtual_shape: tuple[int, ...] | None = None,
    ) -> SlicedTensor:
        """Create by applying slicing to a full tensor/DictBatch."""
        if not isinstance(slicing, Slicing):
            virtual_shape = virtual_shape or _get_shape(value)
            slicing = Slicing(slicing, virtual_shape)
        sliced = slicing.apply(_as_indexable(value))
        return cls(inner_value=sliced, filter=slicing)

    def to(self, *args, **kwargs) -> SlicedTensor:
        return SlicedTensor(
            inner_value=self._inner_value.to(*args, **kwargs),
            filter=self._filter.to(*args, **kwargs),
        )


# ============================================================================
# MaskedTensor
# ============================================================================


@define
class MaskedTensor(FilteredTensorBase[Mask]):
    """A tensor/DictBatch viewed through a Mask filter."""

    _inner_value: Wrappable = field()
    _filter: Mask = field(alias="filter")

    @property
    def inner_value(self) -> Wrappable:
        return self._inner_value

    @property
    def filter(self) -> Mask:
        return self._filter

    # Convenient accessors
    @property
    def mask(self) -> Mask:
        return self._filter

    @property
    def mask_tensor(self) -> Tensor:
        return self._filter.mask

    @property
    def n_selected(self) -> int:
        return self._filter.n_selected

    def __attrs_post_init__(self):
        actual = _get_shape(self._inner_value)
        if actual[0] != self._filter.n_selected:
            raise ValueError(
                f"Inner has {actual[0]} elements, mask selects {self._filter.n_selected}"
            )

    @classmethod
    def from_unmasked(
        cls,
        value: Wrappable,
        mask: Mask | Tensor,
    ) -> MaskedTensor:
        """Create by applying mask to a full tensor/DictBatch."""
        if not isinstance(mask, Mask):
            mask = Mask(mask)
        masked = mask.apply(_as_indexable(value))
        return cls(inner_value=masked, filter=mask)

    def to(self, *args, **kwargs) -> MaskedTensor:
        return MaskedTensor(
            inner_value=self._inner_value.to(*args, **kwargs),
            filter=self._filter.to(*args, **kwargs),
        )

    def refine(self, additional_mask: Mask | Tensor) -> MaskedTensor:
        """
        Apply additional mask in the inner (already-masked) space.
        Returns new MaskedTensor with composed mask.
        """
        if not isinstance(additional_mask, Mask):
            additional_mask = Mask(additional_mask)

        # Apply to inner value
        new_inner = additional_mask.apply(_as_indexable(self._inner_value))

        # Compose masks
        new_mask = self._filter.compose_inner(additional_mask)

        return MaskedTensor(inner_value=new_inner, filter=new_mask)

    def filter_nonzero(self) -> MaskedTensor:
        """
        Refine to only include positions where inner value is nonzero.

        For sparse tensors, uses index presence.
        For dense, uses value > 0.
        """
        inner = _as_indexable(self._inner_value)

        if isinstance(inner, DictBatch):
            raise TypeError("filter_nonzero not supported for DictBatch")

        if inner.is_sparse:
            inner = inner.coalesce()
            active = torch.zeros(self.n_selected, dtype=torch.bool, device=inner.device)
            active[inner.indices()[0].unique()] = True
        else:
            active = inner != 0
            while active.ndim > 1:
                active = active.any(dim=-1)

        return self.refine(Mask(active))


# ============================================================================
# Utilities
# ============================================================================


def _get_shape(v: Wrappable) -> tuple[int, ...]:
    match v:
        case Tensor():
            return tuple(v.shape)
        case FilteredTensorBase():
            return v.inner_shape
        case _:  # DictBatch
            return (v.batch_size,)


def _as_indexable(v: Wrappable) -> Indexable:
    """Get something we can actually index into."""
    match v:
        case Tensor() | DictBatch():
            return v
        case FilteredTensorBase():
            return _as_indexable(v.inner_value)


def _make_empty_like(template: Indexable, shape: tuple[int, ...]) -> Indexable:
    """Create empty tensor/DictBatch with given shape, matching template's properties."""
    match template:
        case Tensor():
            return torch.zeros(
                shape + template.shape[len(shape) :],
                dtype=template.dtype,
                device=template.device,
            )
        case DictBatch():
            # Create DictBatch with same structure but new batch size
            new_batch_size = shape[0] if shape else template.batch_size
            return template.empty_like(batch_size=new_batch_size)


# ============================================================================
# Composition / factory helpers
# ============================================================================


type Filter = FilterBase


def chain(value: Wrappable, *filters: Filter) -> SlicedTensor | MaskedTensor:
    """
    Apply a sequence of filters to a value.

    chain(tensor, slicing1, mask1, slicing2) builds nested structure:
        SlicedTensor(MaskedTensor(SlicedTensor(tensor, slicing1), mask1), slicing2)
    """
    if not filters:
        raise ValueError("Need at least one filter")

    result: Wrappable = value
    for f in filters:
        indexable = _as_indexable(result)
        match f:
            case Slicing():
                result = SlicedTensor(inner_value=f.apply(indexable), filter=f)
            case Mask():
                result = MaskedTensor(inner_value=f.apply(indexable), filter=f)
            case _:
                raise TypeError(f"Unknown filter type: {type(f)}")

    assert isinstance(result, (SlicedTensor, MaskedTensor))
    return result


@overload
def filtered(
    value: Wrappable,
    slicing: Slicing | tuple[slice | int | None, ...],
    mask: None = None,
) -> SlicedTensor: ...


@overload
def filtered(
    value: Wrappable,
    slicing: None,
    mask: Mask | Tensor,
) -> MaskedTensor: ...


@overload
def filtered(
    value: Wrappable,
    slicing: Slicing | tuple[slice | int | None, ...],
    mask: Mask | Tensor,
) -> MaskedTensor: ...


def filtered(
    value: Wrappable,
    slicing: Slicing | tuple[slice | int | None, ...] | None = None,
    mask: Mask | Tensor | None = None,
) -> SlicedTensor | MaskedTensor:
    """
    Convenience constructor: slice then mask.

    Equivalent to original FilteredTensor.from_unmasked_value behavior.
    """
    if slicing is None and mask is None:
        raise ValueError("Need slicing or mask or both")

    filters: list[Filter] = []

    if slicing is not None:
        if not isinstance(slicing, Slicing):
            slicing = Slicing(slicing, _get_shape(value))
        filters.append(slicing)

    if mask is not None:
        if not isinstance(mask, Mask):
            mask = Mask(mask)
        filters.append(mask)

    return chain(value, *filters)


# ============================================================================
# Demo
# ============================================================================


def _demo():
    print("=== Basic Tensor Demo ===")
    x = torch.arange(5000).reshape(100, 50).float()

    slicing = Slicing(
        slices=(slice(10, 60, 2), slice(None, 25)),
        virtual_shape=(100, 50),
    )
    print(f"Slicing: {slicing}")

    mask_tensor = torch.rand(25, 25) > 0.7
    mask = Mask(mask_tensor)
    print(f"Mask: {mask}")

    ft = filtered(x, slicing, mask)
    print(f"FilteredTensor: {ft}")
    print(f"Root value shape: {ft.root_value.shape}")

    # Test writeat
    target = torch.zeros_like(x)
    ft.full_writeat(target)
    print(f"Wrote {(target != 0).sum().item()} nonzero values")

    # Test to_dense round-trip
    dense = ft.to_dense()
    print(f"to_dense shape: {dense.shape}")

    # Verify round-trip
    refiltered = filtered(dense, slicing, mask)
    match = (
        _as_indexable(refiltered.inner_value) == _as_indexable(ft.inner_value)
    ).all()
    print(f"Round-trip matches: {match}")

    print("\n=== Index Conversion Demo ===")
    if ft.n_selected > 2:
        inner_indices = torch.tensor([[0, 1, 2], [0, 0, 0]])
        outer = ft.full_externalize(inner_indices)
        print(f"Externalized: {inner_indices.shape} -> {outer.shape}")

        back, valid = ft.full_internalize(outer)
        print(f"Round-trip valid: {valid.all()}")

    print("\n=== Mask Refinement Demo ===")
    mt = MaskedTensor.from_unmasked(
        torch.arange(100).float(),
        torch.rand(100) > 0.5,
    )
    print(f"Original: {mt.n_selected} selected")

    refined = mt.refine(torch.rand(mt.n_selected) > 0.5)
    print(f"Refined: {refined.n_selected} selected")

    print("\n=== DictBatch Demo (mock) ===")

    # Mock DictBatch-like behavior
    class MockDictBatch:
        def __init__(self, data: dict[str, Tensor]):
            self._data = data
            self._batch_size = next(iter(data.values())).shape[0]

        @property
        def batch_size(self) -> int:
            return self._batch_size

        def __len__(self) -> int:
            return self._batch_size

        def __getitem__(self, idx):
            return MockDictBatch({k: v[idx] for k, v in self._data.items()})

        def __setitem__(self, idx, value):
            for k in self._data:
                self._data[k][idx] = value._data[k]

        def keys(self):
            return self._data.keys()

        def __contains__(self, key):
            return key in self._data

        def to(self, *args, **kwargs):
            return MockDictBatch(
                {k: v.to(*args, **kwargs) for k, v in self._data.items()}
            )

        def empty_like(self, batch_size: int):
            return MockDictBatch(
                {
                    k: torch.zeros(
                        batch_size, *v.shape[1:], dtype=v.dtype, device=v.device
                    )
                    for k, v in self._data.items()
                }
            )

    # This would work with real DictBatch
    db = MockDictBatch(
        {
            "features": torch.randn(100, 32),
            "labels": torch.randint(0, 10, (100,)),
        }
    )

    batch_mask = Mask(torch.rand(100) > 0.7)
    print(f"Batch mask selects {batch_mask.n_selected} of 100")

    # Note: would need to register MockDictBatch with the match statements
    # or use real DictBatch for this to work


if __name__ == "__main__":
    _demo()
