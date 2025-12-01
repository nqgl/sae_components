from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Self, overload

import torch
from attrs import define, field
from torch import Tensor

from saeco.data.dict_batch.dict_batch import DictBatch
from saeco.evaluation.named_filter import NamedFilter

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
            case slice(start=start, stop=stop, step=step):
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
    def inner_shape_prefix(self) -> tuple[int, ...]:
        """
        Shape prefix of the inner tensor.

        This is the prefix of dimensions that this filter transforms.
        The full shape of filtered data may have additional trailing dims.
        """
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
            value: Tensor/DictBatch of filtered_shape_prefix containing values to write
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
        return (
            f"{type(self).__name__}({self.virtual_shape} -> {self.inner_shape_prefix})"
        )


# ============================================================================
# Slicing
# ============================================================================
def slices_converter(
    slices: Sequence[slice | int | None],
) -> tuple[slice | int | None, ...]:
    slices = tuple(slices)
    while len(slices) > 0 and (slices[-1] is None or slices[-1] == slice(None)):
        slices = slices[:-1]
    return slices


@define
class Slicing(FilterBase):
    _slices: tuple[slice | int | None, ...] = field(
        # converter=Converter(slices_converter),
        alias="slices",
    )
    _virtual_shape: tuple[int, ...] = field(alias="virtual_shape")
    """
    Describes a slicing operation on a tensor of known shape.

    For DictBatch, slicing is applied to all fields uniformly.
    Only batch-dimension (dim 0) slicing is well-defined for DictBatch.
    """

    @cached_property
    def slices(self) -> tuple[slice | int | None, ...]:
        return slices_converter(self._slices)

    def __getitem__(self, key: slice):
        return self.__class__(
            slices=self.slices[key], virtual_shape=self.virtual_shape[key]
        )

    def __attrs_post_init__(self):
        if len(self.slices) > len(self.virtual_shape):
            raise ValueError("Sliced dimensions of unknown shape")

    @property
    def virtual_shape(self) -> tuple[int, ...]:
        return self._virtual_shape[: len(self.slices)]

    @property
    def inner_shape_prefix(self) -> tuple[int, ...]:
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

        if hasattr(target, "__setitem__"):
            target[batch_sl] = value
        else:
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
                case slice(start=start, stop=stop, step=step):
                    starts.append(start or 0)
                    steps.append(step or 1)
                    stops.append(stop if stop is not None else self._virtual_shape[i])

        valid = torch.ones(indices.shape[1], dtype=torch.bool, device=device)

        for dim_idx, expected in int_dims.items():
            if dim_idx < indices.shape[0]:
                valid &= indices[dim_idx] == expected

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
                case slice(start=start, step=step):
                    starts.append(start or 0)
                    steps.append(step or 1)

        outer = indices.clone()
        if starts:
            starts_t = torch.tensor(starts, device=device)
            steps_t = torch.tensor(steps, device=device)
            n = min(len(starts), indices.shape[0])
            outer[:n] = indices[:n] * steps_t[:n, None] + starts_t[:n, None]

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
        return self


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
        return self.mask.nonzero()

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
    def inner_shape_prefix(self) -> tuple[int, ...]:
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
        if not isinstance(value, DictBatch):
            raise TypeError(f"Expected DictBatch value, got {type(value)}")
        if not self.is_batch_mask:
            raise ValueError("DictBatch requires 1D mask over batch dimension")

        if hasattr(target, "__setitem__") and hasattr(target, "keys"):
            try:
                target[self.mask] = value
                return
            except (TypeError, NotImplementedError):
                pass

        for key in value.keys():
            if key in target:
                target[key][self.mask] = value[key]

    def internalize_indices(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        assert indices.ndim == 2
        device = indices.device
        mask_ndim = self.mask.ndim

        shape_t = torch.tensor(self.mask.shape, device=device)
        mask_indices = indices[:mask_ndim]

        valid = (mask_indices >= 0).all(dim=0)
        valid &= (mask_indices < shape_t[:, None]).all(dim=0)

        strides = self._strides(device)
        flat_idx = (mask_indices * strides[:, None]).sum(dim=0)
        flat_idx_clamped = flat_idx.clamp(0, self.mask.numel() - 1)

        valid &= self.mask.flatten()[flat_idx_clamped]

        inner_first = self._inverse_map.to(device)[flat_idx_clamped]
        inner_indices = torch.cat([inner_first[None], indices[mask_ndim:]])
        assert not (valid & (inner_indices == -1)).any()

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

    def to(self, *args, **kwargs) -> Self:
        filter_kwargs = kwargs
        if "dtype" in kwargs:
            filter_kwargs = {**kwargs, "dtype": torch.bool}
        filter_args = args
        for i, arg in enumerate(args):
            if isinstance(arg, torch.dtype):
                filter_args = filter_args[:i] + (filter_kwargs,) + filter_args[i + 1 :]

        new_mask = self.mask.to(*filter_args, **filter_kwargs)
        if new_mask is self.mask:
            return self
        return self.__class__(new_mask)

    def __and__(self, other: "Mask") -> Self:
        if self.mask.shape != other.mask.shape:
            raise ValueError(f"Shape mismatch: {self.mask.shape} vs {other.mask.shape}")
        return self.__class__(self.mask & other.mask)

    def __or__(self, other: "Mask") -> Self:
        if self.mask.shape != other.mask.shape:
            raise ValueError(f"Shape mismatch: {self.mask.shape} vs {other.mask.shape}")
        return self.__class__(self.mask | other.mask)

    def __invert__(self) -> Self:
        return self.__class__(~self.mask)

    def __xor__(self, other: "Mask") -> Self:
        if self.mask.shape != other.mask.shape:
            raise ValueError(f"Shape mismatch: {self.mask.shape} vs {other.mask.shape}")
        return self.__class__(self.mask ^ other.mask)

    def compose_inner(self, inner_mask: "Mask") -> Self:
        """
        Compose with a mask in the inner (already-masked) space.
        """
        if inner_mask.mask.shape[0] != self._n_selected:
            raise ValueError(
                f"Inner mask size {inner_mask.mask.shape[0]} doesn't match "
                f"current selection count {self._n_selected}"
            )

        new_flat = torch.zeros(
            self.mask.numel(),
            dtype=torch.bool,
            device=self.mask.device,
        )
        current_true_flat = self.mask.flatten().nonzero().squeeze(-1)
        new_flat[current_true_flat[inner_mask.mask]] = True

        return self.__class__(new_flat.view(self.mask.shape))


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
    def filtered_shape_prefix(self) -> tuple[int, ...]:
        return self.filter.inner_shape_prefix

    @property
    def root_value(self) -> Indexable:
        """Recursively unwrap to innermost tensor/DictBatch."""
        v = self.inner_value
        while isinstance(v, FilteredTensorBase):
            v = v.inner_value
        return v

    @property
    def is_dictbatch(self) -> bool:
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
        if value is None:
            value = _as_indexable(self.inner_value)
        self.filter.writeat(target, value)

    def full_writeat(self, target: Indexable, value: Indexable | None = None) -> None:
        if value is None:
            value = _as_indexable(self.inner_value)

        if isinstance(self.inner_value, FilteredTensorBase):
            intermediate = _make_empty_like(target, self.filter.inner_shape_prefix)
            self.inner_value.full_writeat(intermediate, value)
            self.filter.writeat(target, intermediate)
        else:
            self.filter.writeat(target, value)

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

    @property
    def slicing(self) -> Slicing:
        return self._filter

    @property
    def slices(self) -> tuple[slice | int | None, ...]:
        return self._filter.slices

    @property
    def num_indexed(self) -> int:
        return sum(isinstance(s, int) for s in self.slices)

    def __attrs_post_init__(self):
        actual = _get_shape(self._inner_value)
        expected = self._filter.inner_shape_prefix
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
    ) -> Self:
        if not isinstance(slicing, Slicing):
            virtual_shape = virtual_shape or _get_shape(value)
            slicing = Slicing(slices=slicing, virtual_shape=virtual_shape)
        sliced = slicing.apply(_as_indexable(value))
        return cls(inner_value=sliced, filter=slicing)

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
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
    def virtual_shape(self) -> tuple[int, ...]:
        return self._filter.virtual_shape + _get_outer_shape(self._inner_value)[1:]

    @property
    def inner_value(self) -> Wrappable:
        return self._inner_value

    @property
    def filter(self) -> Mask:
        return self._filter

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
    ) -> Self:
        if not isinstance(mask, Mask):
            mask = Mask(mask)
        masked = mask.apply(_as_indexable(value))
        return cls(inner_value=masked, filter=mask)

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            inner_value=self._inner_value.to(*args, **kwargs),
            filter=self._filter.to(*args, **kwargs),
        )

    def refine(self, additional_mask: Mask | Tensor) -> Self:
        if not isinstance(additional_mask, Mask):
            additional_mask = Mask(additional_mask)

        new_inner = additional_mask.apply(_as_indexable(self._inner_value))
        new_mask = self._filter.compose_inner(additional_mask)

        return self.__class__(inner_value=new_inner, filter=new_mask)

    def filter_nonzero(self) -> Self:
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
# FilteredTensor - Compatibility class (slice then mask)
# ============================================================================
@define
class FilterProxy:
    slices: Sequence[slice | int | None] = field()
    mask: Tensor | None = field()
    shape: tuple[int, ...] = field()
    orig: "FilteredTensor | None" = field(default=None)

    @property
    def virtual_shape(self) -> tuple[int, ...]: ...
    def writeat(self, target: Tensor, value: Tensor) -> None:
        if self.orig is not None:
            return self.orig.writeat(target, value)
        raise NotImplementedError("writeat not implemented for FilterProxy")


@define
class FilteredTensor:
    """
    Represents a virtual larger tensor via slicing + masking.

    This is a compatibility class that replicates the original FilteredTensor
    behavior using the modular Slicing and Mask filters.

    Conceptually:
        value = mask.apply(slicing.apply(virtual_tensor))
    """

    value: Tensor | DictBatch = field()
    slicing: Slicing | None = field(default=None)
    mask: Mask | None = field(default=None)

    # Cached combined filter for index operations
    _nested: SlicedTensor | MaskedTensor | None = field(init=False, repr=False)

    @_nested.default
    def _build_nested(self) -> SlicedTensor | MaskedTensor | None:
        """Build nested filter structure for index operations."""
        if self.slicing is None and self.mask is None:
            return None

        if self.slicing is not None and self.mask is not None:
            # SlicedTensor containing MaskedTensor's inner value reference
            sliced_shape = self.slicing.inner_shape_prefix
            inner_masked = MaskedTensor(inner_value=self.value, filter=self.mask)
            return SlicedTensor(inner_value=inner_masked, filter=self.slicing)
        elif self.slicing is not None:
            return SlicedTensor(inner_value=self.value, filter=self.slicing)
        else:
            assert self.mask is not None
            return MaskedTensor(inner_value=self.value, filter=self.mask)

    @cached_property
    def filter(self) -> FilterProxy:
        return FilterProxy(
            mask=self.mask.mask if self.mask is not None else None,
            slices=self.slicing.slices if self.slicing is not None else None,
            shape=self.virtual_shape,
            orig=self,
        )

    def __attrs_post_init__(self):
        if self.slicing is None and self.mask is None:
            raise ValueError("FilteredTensor requires at least slicing or mask")

        # Validate value shape matches expected filtered shape
        expected = self._compute_filtered_shape_prefix()
        actual = _get_shape(self.value)
        if actual[0] != expected[0]:
            raise ValueError(
                f"Value shape {actual} doesn't match expected filtered shape prefix {expected}"
            )

    def _compute_filtered_shape_prefix(self) -> tuple[int, ...]:
        """Compute the expected shape prefix after slicing and masking."""
        if self.mask is not None:
            return self.mask.inner_shape_prefix
        elif self.slicing is not None:
            return self.slicing.inner_shape_prefix
        return ()

        m = self.mask.virtual_shape if self.mask is not None else ()
        s = self.slicing.virtual_shape if self.slicing is not None else ()
        m = m + (0,) * (len(s) - len(m))
        s = s + (0,) * (len(m) - len(s))
        return tuple(max(i, j) for i, j in zip(m, s, strict=True))
        return ()

    @property
    def virtual_shape(self) -> tuple[int, ...]:
        """Shape of the conceptual full tensor."""
        shape = self.mask.virtual_shape if self.mask is not None else ()
        if self.slicing is not None:
            shape = (
                self.slicing.virtual_shape
                + shape[len(self.slicing.virtual_shape) - self.slicing.ndim_removed :]
            )
        return tuple(shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Alias for virtual_shape (compatibility)."""
        base = self.virtual_shape

        # Add trailing dims from value
        prefix_len = len(self._compute_filtered_shape_prefix())
        return base + _get_shape(self.value)[prefix_len:]

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.value, Tensor) and self.value.is_sparse

    @property
    def device(self) -> torch.device:
        if isinstance(self.value, Tensor):
            return self.value.device
        # DictBatch - get device from first tensor
        for v in self.value.values():
            return v.device
        raise ValueError("Cannot determine device of empty DictBatch")

    @property
    def n_selected(self) -> int:
        """Number of elements selected by the mask."""
        if self.mask is not None:
            return self.mask.n_selected
        if self.slicing is not None:
            return self.slicing.inner_shape_prefix[0]
        return _get_shape(self.value)[0]

    # ---- Factory methods ----

    @classmethod
    def from_value_and_mask(
        cls,
        value: Tensor | DictBatch,
        mask_obj: Tensor | Mask | None,
    ) -> Self:
        """
        Construct from a value that already corresponds to the mask.

        The value should already be masked: value.shape[0] == mask.sum()
        """
        if mask_obj is None:
            # Create trivial full mask
            batch_size = _get_shape(value)[0]
            device = value.device if isinstance(value, Tensor) else _get_device(value)
            mask_obj = Mask(torch.ones(batch_size, dtype=torch.bool, device=device))
        elif isinstance(mask_obj, Tensor):
            mask_obj = Mask(mask_obj)

        return cls(value=value, mask=mask_obj)

    @classmethod
    def from_unmasked_value(
        cls,
        value: Tensor | DictBatch,
        slicing: Slicing | tuple[slice | int | None, ...] | None = None,
        mask: Mask | Tensor | None = None,
        filter_obj: FilterProxy | NamedFilter | Tensor | None = None,
        presliced: bool = False,
    ) -> Self:
        """
        Construct from an unmasked value and filters.

        Args:
            value: The source tensor/DictBatch
            slicing: Optional slicing to apply
            mask: Optional mask to apply (after slicing)
            presliced: If True, value already has slicing applied
        """
        virtual_shape = _get_shape(value)
        if filter_obj is not None:
            assert mask is None
            if isinstance(filter_obj, NamedFilter):
                mask = filter_obj.filter
            elif isinstance(filter_obj, FilterProxy):
                slices = filter_obj.slices
                if filter_obj.shape is not None:
                    if len(filter_obj.shape) >= len(virtual_shape):
                        assert filter_obj.shape[: len(virtual_shape)] == virtual_shape
                        virtual_shape = filter_obj.shape
                while slices and (slices[-1] is None or slices[-1] == slice(None)):
                    slices = slices[:-1]
                if slicing is not None:
                    if slices:
                        raise ValueError("Duplicate slicing provided")
                else:
                    slicing = tuple(filter_obj.slices)
                mask = filter_obj.mask
            else:
                assert isinstance(filter_obj, Tensor)
                mask = filter_obj
        # Normalize mask
        if mask is not None and not isinstance(mask, Mask):
            mask = Mask(mask)
        # Normalize slicing
        if slicing is not None and not isinstance(slicing, Slicing):
            while slicing and (slicing[-1] is None or slicing[-1] == slice(None)):
                slicing = slicing[:-1]

            ndim_sliced = len(slicing)
            slicing = Slicing(
                slices=slicing,
                virtual_shape=virtual_shape[:ndim_sliced],
            )

        # Apply slicing if needed
        if slicing is not None and not presliced:
            value = slicing.apply(value)

        # Apply mask if needed
        if mask is not None:
            value = mask.apply(value)

        return cls(value=value, slicing=slicing, mask=mask)

    # ---- Index operations ----

    def internalize_indices(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        """
        Convert virtual-space indices to value-space indices.

        Returns (inner_indices, valid_mask).
        """
        if self._nested is not None:
            return self._nested.full_internalize(indices)
        return indices, torch.ones(
            indices.shape[1], dtype=torch.bool, device=indices.device
        )

    def externalize_indices(self, indices: Tensor) -> Tensor:
        """Convert value-space indices back to virtual-space indices."""
        if self._nested is not None:
            return self._nested.full_externalize(indices)
        return indices

    def index_where_valid(self, indices: Tensor) -> tuple[Tensor | DictBatch, Tensor]:
        """
        Index the value, returning results and validity mask.

        Args:
            indices: (ndim, n_points) coordinates in virtual space

        Returns:
            (values_at_valid_indices, valid_mask)
        """
        inner_idx, valid = self.internalize_indices(indices)
        inner_idx_valid = inner_idx[:, valid]

        if isinstance(self.value, DictBatch):
            result = self.value[inner_idx_valid[0]]
        elif self.is_sparse:
            result = self.value.index_select(0, inner_idx_valid[0])
        else:
            result = self.value[tuple(inner_idx_valid)]

        return result, valid

    def index_select(self, index: Tensor, dim: int = 0) -> Tensor | DictBatch:
        """
        Select along dimension using virtual-space indices.

        Only dim=0 is currently supported.
        """
        if dim != 0:
            raise NotImplementedError("Only dim=0 is implemented")

        assert index.ndim == 1
        indices_2d = index.unsqueeze(0)

        inner_idx, valid = self.internalize_indices(indices_2d)
        if not valid.all():
            raise IndexError("Some indices are outside the filtered region")

        inner_index = inner_idx.squeeze(0)

        if isinstance(self.value, DictBatch):
            return self.value[inner_index]
        return self.value.index_select(dim, inner_index)

    # ---- Mask operations ----

    def mask_by_other(
        self,
        other: Mask | Tensor | Self,
        return_ft: bool = False,
        presliced: bool = False,
        value_like: bool = False,
    ) -> Tensor | DictBatch | Self:
        """
        Apply additional mask on top of this FilteredTensor.

        Args:
            other: Mask to apply
            return_ft: If True, return a new FilteredTensor
            presliced: If True, mask already matches sliced view
            value_like: If True, mask is in value space (shape matches self.value)
        """
        if isinstance(other, FilteredTensor):
            other = other.mask

        if isinstance(other, Tensor):
            other = Mask(other)

        if other is None:
            return self if return_ft else self.value

        # Get the mask tensor in the right space
        if value_like:
            # Mask is in value space
            filtered_mask = other.mask
        elif presliced:
            # Mask is in sliced space (after slicing, before masking)
            if self.mask is not None:
                filtered_mask = other.mask[self.mask.mask]
            else:
                filtered_mask = other.mask
        else:
            # Mask is in virtual space - need to slice then apply current mask
            if self.slicing is not None:
                sliced_mask = self.slicing.apply(other.mask)
            else:
                sliced_mask = other.mask

            if self.mask is not None:
                filtered_mask = sliced_mask[self.mask.mask]
            else:
                filtered_mask = sliced_mask

        # Apply to value
        if isinstance(self.value, DictBatch):
            new_value = self.value[filtered_mask]
        elif self.is_sparse:
            new_value = _sparse_mask(self.value.coalesce(), filtered_mask)
        else:
            new_value = self.value[filtered_mask]

        if not return_ft:
            return new_value

        # Build new composed mask in virtual space
        if self.mask is None:
            if value_like:
                new_mask = other
            else:
                if self.slicing is not None:
                    sliced_mask = self.slicing.apply(other.mask)
                else:
                    sliced_mask = other.mask
                new_mask = Mask(sliced_mask)
        else:
            if value_like:
                new_mask = self.mask.compose_inner(other)
            else:
                if self.slicing is not None:
                    sliced_mask = self.slicing.apply(other.mask)
                else:
                    sliced_mask = other.mask
                new_mask = self.mask & Mask(sliced_mask)

        return self.__class__(
            value=new_value,
            slicing=self.slicing,
            mask=new_mask,
        )

    def filter_inactive_docs(self) -> Self:
        """
        Refine to only include positions where value is nonzero.
        """
        if isinstance(self.value, DictBatch):
            raise TypeError("filter_inactive_docs not supported for DictBatch")

        if self.is_sparse:
            v = self.value.coalesce()
            active = torch.zeros(
                _get_shape(self.value)[0],
                dtype=torch.bool,
                device=v.device,
            )
            active[v.indices()[0].unique()] = True
        else:
            active = self.value != 0
            while active.ndim > 1:
                active = active.any(dim=-1)

        return self.mask_by_other(
            Mask(active),
            return_ft=True,
            value_like=True,
        )

    def to_filtered_like_self(
        self,
        t: Tensor,
        presliced: bool = True,
        premasked: bool = True,
        ndim: int | None = None,
    ) -> Self:
        """
        Wrap tensor t with the same filters as self.

        Args:
            t: Tensor to wrap
            presliced: If True, t already has slicing applied
            premasked: If True, t already has mask applied
        """
        if premasked and not presliced:
            raise ValueError("Cannot be masked and not sliced")

        if not presliced and self.slicing is not None:
            t = self.slicing.apply(t)

        if not premasked and self.mask is not None:
            t = self.mask.apply(t)
        if ndim is not None:
            slicing = self.slicing[:ndim]
        else:
            slicing = self.slicing
        return self.__class__(
            value=t,
            slicing=slicing,
            mask=self.mask,
        )

    # ---- Materialization ----

    def to_dense_unfiltered(self, fill_value: float = 0.0) -> Tensor:
        """
        Materialize the full virtual tensor.

        Unselected positions are filled with fill_value.
        """
        if isinstance(self.value, DictBatch):
            raise TypeError("to_dense not supported for DictBatch")

        dense_value = self.value.to_dense() if self.is_sparse else self.value

        result = torch.full(
            self.shape,
            fill_value,
            dtype=dense_value.dtype,
            device=dense_value.device,
        )

        self.writeat(result, dense_value)
        return result

    def to_dense(self) -> Self:
        return self.to_filtered_like_self(
            self.value.to_dense(), premasked=True, presliced=True
        )

    def writeat(
        self, target: Tensor | DictBatch, value: Tensor | DictBatch | None = None
    ) -> None:
        """
        Write value into target at filtered positions.
        """
        if value is None:
            value = self.value

        if self.slicing is not None:
            sliced = self.slicing.apply(target)
        else:
            sliced = target
        assert isinstance(sliced, Tensor), "sliced must be a Tensor"
        assert sliced._is_view() or sliced is target
        if self.mask is not None:
            # make 'do fn to x' construct if need to do to DB as well
            sliced[self.mask.mask] = value
        else:
            sliced[:] = value

    # ---- Sparse tensor operations ----

    def indices(self) -> Tensor:
        """
        Get indices of nonzero elements in virtual space.

        Only valid for sparse tensor values.
        """
        if not self.is_sparse:
            raise TypeError("indices() only valid for sparse tensors")

        inner_indices = self.value.coalesce().indices()
        return self.externalize_indices(inner_indices)

    def values(self) -> Tensor:
        """
        Get values of nonzero elements.

        Only valid for sparse tensor values.
        """
        if not self.is_sparse:
            raise TypeError("values() only valid for sparse tensors")

        return self.value.coalesce().values()

    def nonzero(self) -> Tensor:
        """
        Get indices of nonzero elements in virtual space.

        Works for both dense and sparse tensors.
        """
        if isinstance(self.value, DictBatch):
            raise TypeError("nonzero() not supported for DictBatch")

        if self.is_sparse:
            return self.indices()

        inner_nz = self.value.nonzero().T
        return self.externalize_indices(inner_nz).T

    # ---- Device operations ----

    def cuda(self) -> Self:
        """Move to CUDA device."""
        return self.to(device="cuda")

    def to(self, *args, **kwargs) -> Self:
        """Move to specified device/dtype."""
        new_value = self.value.to(*args, **kwargs)
        new_slicing = self.slicing.to(*args, **kwargs) if self.slicing else None
        new_mask = self.mask.to(*args, **kwargs) if self.mask else None

        return self.__class__(
            value=new_value,
            slicing=new_slicing,
            mask=new_mask,
        )

    def __repr__(self) -> str:
        return (
            f"FilteredTensor(shape={self.virtual_shape}, "
            f"n_selected={self.n_selected}, "
            f"slicing={self.slicing is not None}, "
            f"mask={self.mask is not None})"
        )

    def to_sparse_unfiltered(self) -> Tensor:
        # TODO just deprecate this? not used and doesn't make sense for dictbatched FTs
        """
        Materialise the full virtual tensor as a sparse COO tensor.

        Only defined when the underlying value is a sparse tensor.
        """
        if not self.is_sparse or isinstance(self.value, DictBatch):
            raise TypeError(
                "to_sparse_unfiltered is only defined for sparse tensor values"
            )
        return torch.sparse_coo_tensor(
            indices=self.indices(),
            values=self.values(),
            size=self.shape,
            dtype=self.value.dtype,
            device=self.value.device,
        )

    def __getitem__(self, key: Tensor):
        if not isinstance(key, Tensor) or key.dtype != torch.long or key.ndim != 2:
            raise NotImplementedError(
                "Indexing FilteredTensor is only implemented for 2D integer tensors"
            )
        ids, mask = self.internalize_indices(key)
        return self.value[ids.unbind()]


# ============================================================================
# Utilities
# ============================================================================


def _get_shape(v: Wrappable) -> tuple[int, ...]:
    match v:
        case Tensor():
            return tuple(v.shape)
        case FilteredTensorBase():
            return v.virtual_shape
        case _:  # DictBatch
            return (v.batch_size,)


def _get_outer_shape(v: Wrappable) -> tuple[int, ...]:
    """Shape of the actual stored data."""
    match v:
        case Tensor():
            return tuple(v.shape)
        case FilteredTensorBase():
            return v.filtered_shape_prefix
        case _:  # DictBatch
            return (v.batch_size,)


def _get_inner_shape(v: Wrappable) -> tuple[int, ...]:
    """Shape this object claims to represent."""
    match v:
        case Tensor():
            return tuple(v.shape)
        case FilteredTensorBase():
            return v.virtual_shape
        case _:  # DictBatch
            return (v.batch_size,)


def _get_device(v: DictBatch) -> torch.device:
    """Get device from DictBatch."""
    for tensor in v.values():
        return tensor.device
    return torch.device("cpu")


def _as_indexable(v: Wrappable) -> Indexable:
    """Get something we can actually index into."""
    match v:
        case Tensor() | DictBatch():
            return v
        case FilteredTensorBase():
            return _as_indexable(v.inner_value)


def _make_empty_like(template: Indexable, shape: tuple[int, ...]) -> Indexable:
    """Create empty tensor/DictBatch with given shape prefix."""
    match template:
        case Tensor():
            return torch.zeros(
                shape + template.shape[len(shape) :],
                dtype=template.dtype,
                device=template.device,
            )
        case DictBatch():
            new_batch_size = shape[0] if shape else template.batch_size
            return template.empty_like(batch_size=new_batch_size)


# ============================================================================
# Composition / factory helpers
# ============================================================================


type Filter = FilterBase


def chain(value: Wrappable, *filters: Filter) -> SlicedTensor | MaskedTensor:
    """Apply a sequence of filters to a value."""
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
    """Convenience constructor: slice then mask."""
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


def _filter(
    slices: Sequence[slice | int | None] = field(),
    mask: Tensor | None = field(),
    shape: tuple[int, ...] = field(),
):
    return FilterProxy(slices=slices, mask=mask, shape=shape)


Filter = _filter

# ============================================================================
# Demo
# ============================================================================


def _demo():
    print("=== FilteredTensor Compatibility Demo ===\n")

    # Create test tensor
    x = torch.arange(5000).reshape(100, 50).float()
    print(f"Source tensor: {x.shape}")

    # Test from_unmasked_value with slicing + mask
    slicing = Slicing(
        slices=(slice(10, 60, 2), slice(None, 25)),
        virtual_shape=(100, 50),
    )
    mask_tensor = torch.rand(25, 25) > 0.7
    mask = Mask(mask_tensor)
    x.shape
    slicing.virtual_shape
    slicing.inner_shape_prefix
    slicing.apply(x).shape

    ft = FilteredTensor.from_unmasked_value(x, slicing=slicing, mask=mask)
    print(f"FilteredTensor: {ft}")
    print(f"  virtual_shape: {ft.virtual_shape}")
    print(f"  n_selected: {ft.n_selected}")
    print(f"  value shape: {ft.value.shape}")

    # Test to_dense round-trip
    print("\n--- to_dense round-trip ---")
    dense = ft.to_dense(fill_value=-1)
    print(f"Dense shape: {dense.shape}")

    ft2 = FilteredTensor.from_unmasked_value(dense, slicing=slicing, mask=mask)
    match = (ft2.value == ft.value).all()
    print(f"Round-trip matches: {match}")

    # Test index operations
    print("\n--- Index operations ---")
    virtual_indices = torch.tensor([[12, 14, 16], [0, 2, 4]])  # Some virtual coords
    inner_idx, valid = ft.internalize_indices(virtual_indices)
    print(f"Internalized {valid.sum()}/{len(valid)} valid")

    if valid.any():
        external = ft.externalize_indices(inner_idx[:, valid])
        print(f"Externalized back: shape {external.shape}")

    # Test index_where_valid
    print("\n--- index_where_valid ---")
    values, valid = ft.index_where_valid(virtual_indices)
    print(f"Got {valid.sum()} valid values")

    # Test mask_by_other
    print("\n--- mask_by_other ---")
    additional_mask = torch.rand(ft.n_selected) > 0.5
    ft_refined = ft.mask_by_other(additional_mask, return_ft=True, value_like=True)
    print(f"Refined: {ft.n_selected} -> {ft_refined.n_selected}")

    # Test filter_inactive_docs
    print("\n--- filter_inactive_docs ---")
    sparse_val = torch.zeros(ft.n_selected, 10)
    sparse_val[::2, :5] = 1  # Only half have values
    ft_sparse = FilteredTensor(value=sparse_val, slicing=slicing, mask=mask)
    ft_active = ft_sparse.filter_inactive_docs()
    print(f"Active docs: {ft_sparse.n_selected} -> {ft_active.n_selected}")

    # Test from_value_and_mask
    print("\n--- from_value_and_mask ---")
    some_mask = torch.rand(100) > 0.6
    some_value = torch.randn(some_mask.sum().item(), 32)
    ft_from_mask = FilteredTensor.from_value_and_mask(some_value, some_mask)
    print(f"From value+mask: {ft_from_mask}")

    # Test to_filtered_like_self
    print("\n--- to_filtered_like_self ---")
    new_data = torch.randn_like(ft.value)
    ft_like = ft.to_filtered_like_self(new_data)
    print(
        f"to_filtered_like_self: same shape = {ft_like.value.shape == ft.value.shape}"
    )

    # Test sparse tensor operations
    print("\n--- Sparse operations ---")
    sparse_indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
    sparse_values = torch.tensor([1.0, 2.0, 3.0])
    sparse_t = torch.sparse_coo_tensor(
        sparse_indices, sparse_values, (ft.n_selected, 10)
    )
    ft_sp = FilteredTensor(value=sparse_t, slicing=slicing, mask=mask)
    print(f"Sparse FilteredTensor: is_sparse={ft_sp.is_sparse}")
    print(f"  indices shape: {ft_sp.indices().shape}")
    print(f"  values shape: {ft_sp.values().shape}")

    # Test device operations
    print("\n--- Device operations ---")
    ft_cpu = ft.to(device="cpu")
    print(f"Moved to CPU: device={ft_cpu.device}")

    print("\n=== Demo complete ===")


if __name__ == "__main__":
    _demo()
