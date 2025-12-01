from collections.abc import Sequence
from typing import Any, Literal, Self, overload

import torch
from attrs import Converter, define, field, validators
from torch import Tensor

from saeco.data.dict_batch.dict_batch import DictBatch, DictBatchShapes
from saeco.evaluation.named_filter import NamedFilter

DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"


def assert_isint(x: Any) -> int:
    assert isinstance(x, int)
    return x


def convert(fld):
    def converter_wrapper(fn):
        assert fld.converter is None
        fld.converter = Converter(fn)
        return fn

    return converter_wrapper


def right_expand(t: Tensor, shape: tuple[int, ...]) -> Tensor:
    assert len(shape) >= len(t.shape)
    assert all(
        t.shape[i] == shape[i] or shape[i] == -1 or t.shape[i] == 1
        for i in range(len(t.shape))
    ), f"Shapes do not match for right-expand: {t.shape} and {shape}"
    for _ in range(len(shape) - len(t.shape)):
        t = t.unsqueeze(-1)
    return t.expand(shape)


def slice_shape(
    input_shape: tuple[int, ...] | list[int], slices: Sequence[slice | int | None]
) -> tuple[int, ...]:
    """
    Compute the output shape after applying `slices` as indexing to a tensor
    with shape `input_shape`.
    """
    if isinstance(slices, slice):
        slices = [slices]
    slices = tuple(slices) + (None,) * (len(input_shape) - len(slices))
    out_shape: list[int] = []
    for slc, shape in zip(slices, input_shape, strict=True):
        if isinstance(slc, int):
            # Dimension removed
            continue
        if slc is None:
            out_shape.append(shape)
            continue
        slmin = slc.start or 0
        slmax = min(slc.stop or shape, shape)
        step = slc.step or 1
        out_shape.append((slmax - slmin + step - 1) // step)
    return tuple(out_shape + list(input_shape[len(slices) :]))


def _calculate_ndim_filtered(
    slices: Sequence[slice | int | None], mask: Tensor | None
) -> int:
    nontrivial_slicing = slices
    while len(nontrivial_slicing) > 0 and nontrivial_slicing[-1] is None:
        nontrivial_slicing = nontrivial_slicing[:-1]
    n = len(nontrivial_slicing)
    m = sum(isinstance(i, int) for i in nontrivial_slicing)
    mask_overlap = (n - m) + (mask.ndim if mask is not None else 0)
    return max(n, mask_overlap)


@define
class Filter:
    """
    Describes a combination of slicing + prefix boolean masking on a
    virtual "large" tensor of shape `shape`.

    - First, `slices` is applied to the base tensor.
    - Then, `mask` (a boolean tensor) is applied to the *leading* dimensions
      of that sliced view.

    For `DictBatch` we only ever slice/mask along the batch dimension, so:
    - `len(slices) <= 1`
    - `mask.ndim == 1`
    """

    slices: Sequence[slice | int | None] = field()
    mask: Tensor | None = field()
    shape: tuple[int, ...] = field()

    @property
    def virtual_shape(self) -> tuple[int, ...]:
        if len(self.shape) < self.ndim_filtered():
            raise ValueError(
                f"Shape {self.shape} has less dimensions than the number"
                f" of virtual dimensions {self.ndim_filtered()}"
                ", leaving virtual shape underspecified"
            )
        return self.shape[: self.ndim_filtered()]

    @classmethod
    def construct_from_full_shape(
        cls,
        slices: Sequence[slice | int | None],
        mask: Tensor | None,
        shape: tuple[int, ...],
    ) -> Self:
        num_virt = _calculate_ndim_filtered(slices, mask)
        # if len(shape) > num_virt:
        #     shape = shape[:num_virt]
        #     raise ValueError(
        #         f"Shape {shape} has more dimensions than the number"
        #         f" of virtual dimensions {num_virt}"
        #     )
        return cls(slices=slices, mask=mask, shape=shape[:num_virt])

    def __attrs_post_init__(self):
        while len(self.slices) > 0 and self.slices[-1] is None:
            self.slices = self.slices[:-1]
        sliced = slice_shape(self.virtual_shape, self.slices)
        # assert isinstance(self.mask, Tensor | None)

        # if not isinstance(self.mask, Tensor):
        #     # Treat None / device specifier as "no mask" → full True on first dim.
        #     device = DEVICE_DEFAULT
        #     if isinstance(self.mask, (str, torch.device)):
        #         device = self.mask
        #     mask = torch.ones(1, dtype=torch.bool, device=device)
        #     self.mask = mask.expand(sliced[0])
        if self.mask is not None:
            assert isinstance(self.mask, Tensor)
            if self.mask.dtype is not torch.bool:
                raise TypeError("Filter.mask must be a boolean tensor")
            # if len(sliced) > :
            #     raise ValueError(
            #         f"Mask shape {tuple(self.mask.shape)} is not a prefix of sliced "
            #         f"shape {sliced}"
            # )
            if tuple(self.mask.shape) != sliced[: len(self.mask.shape)]:
                raise ValueError(
                    f"Mask shape {tuple(self.mask.shape)} incompatible with sliced "
                    f"prefix {sliced[: self.mask.ndim]} for shape={self.virtual_shape} and "
                    f"slices={self.slices!r}"
                )

    def ndim_filtered(self) -> int:
        nontrivial_slicing = self.slices
        while len(nontrivial_slicing) > 0 and nontrivial_slicing[-1] is None:
            nontrivial_slicing = nontrivial_slicing[:-1]
        n = len(nontrivial_slicing)
        m = sum(isinstance(i, int) for i in nontrivial_slicing)
        mask_overlap = (n - m) + (self.mask.ndim if self.mask is not None else 0)
        return max(n, mask_overlap)

    def slicing_overhang_ndim(self) -> int:
        nontrivial_slicing = self.slices
        while len(nontrivial_slicing) > 0 and nontrivial_slicing[-1] is None:
            nontrivial_slicing = nontrivial_slicing[:-1]
        index_shift_after_slicing = 0
        n_sliced = len(nontrivial_slicing)
        for slc in nontrivial_slicing:
            if isinstance(slc, int):
                index_shift_after_slicing += 1
            else:
                break
        mask_depth = self.mask.ndim if self.mask is not None else 0
        adj_mask_depth = mask_depth + index_shift_after_slicing
        return n_sliced - adj_mask_depth
        # m = sum(isinstance(i, int) for i in nontrivial_slicing)
        # mask_overlap = (n_sliced - m) + (self.mask.ndim if self.mask is not None else 0)
        # return max(n_sliced, mask_overlap)

    def _inner_shape(self) -> tuple[int, ...]:
        # TODO make this func of arg instead of static
        """
        Shape of `x[self.slicing_tuple][self.mask]` when `x.shape == self.shape`.
        """
        sliced = slice_shape(self.virtual_shape, self.slices)
        if self.mask is None:
            return sliced
        assert tuple(self.mask.shape) == sliced[: self.mask.ndim]
        return (assert_isint(self.mask.sum().item()),) + tuple(sliced[self.mask.ndim :])

    def _calculate_inner_shape(self, virtual_shape: tuple[int, ...]) -> tuple[int, ...]:
        # TODO make this func of arg instead of static
        """
        Shape of `x[self.slicing_tuple][self.mask]` when `x.shape == self.shape`.
        """
        sliced = slice_shape(virtual_shape, self.slices)
        if self.mask is None:
            return sliced
        assert tuple(self.mask.shape) == sliced[: self.mask.ndim]
        return (assert_isint(self.mask.sum().item()),) + tuple(sliced[self.mask.ndim :])

    def _check_shapes(
        self,
        outer_shape: tuple[int, ...],
        inner_shape: tuple[int, ...],
    ) -> None:
        shape = inner_shape
        if self.mask is not None:
            assert assert_isint(self.mask.sum().item()) == inner_shape[0]
            shape = self.mask.shape + inner_shape[1:]
        sliced = slice_shape(outer_shape, self.slices)
        assert sliced == shape

    @overload
    def apply(self, tensor: Tensor) -> Tensor: ...
    @overload
    def apply(self, tensor: DictBatch) -> DictBatch: ...

    def apply(self, tensor: Tensor | DictBatch) -> Tensor | DictBatch:
        """
        Apply both `slices` and `mask` to `tensor`.

        For `DictBatch` this is restricted to the batch dimension.
        """
        sliced = self.slice(tensor)
        return self.apply_mask(sliced)

    def intersect(self, other: "Filter") -> "Filter":
        raise NotImplementedError(
            "Intersect not implemented, need to make slice intersect"
        )

    @property
    def slicing_tuple(self) -> tuple[slice | int, ...]:
        return tuple(
            sl if sl is not None else slice(None, None, None) for sl in self.slices
        )

    @overload
    def apply_mask(self, tensor: Tensor) -> Tensor: ...
    @overload
    def apply_mask(self, tensor: DictBatch) -> DictBatch: ...

    def apply_mask(self, tensor: Tensor | DictBatch) -> Tensor | DictBatch:
        """
        Apply only the boolean mask (no slicing).

        For DictBatch we **only** support masking along the batch (0th) dim.
        """
        if self.mask is None:
            return tensor

        mask = self.mask

        # DictBatch branch: mask is 1D over the batch dimension
        if isinstance(tensor, DictBatch):
            if mask.ndim != 1:
                raise ValueError(
                    f"DictBatch masking expects a 1D mask over the batch dimension, "
                    f"got {mask.ndim}D"
                )
            if len(tensor) != mask.shape[0]:
                raise ValueError(
                    f"Mask length {mask.shape[0]} does not match DictBatch batch size "
                    f"{len(tensor)}"
                )
            # DictBatch.__getitem__(bool_mask) already does per-tensor masking
            return tensor[mask]

        # Tensor branch (unchanged)
        if tensor.is_sparse:
            if not tensor.is_coalesced():
                tensor = tensor.coalesce()
            return index_sparse_with_bool(
                tensor, self.mask
            ).coalesce()  # TODO for dictbatch  case

        return tensor[self.mask]

    def writeat(self, target: Tensor, value: Tensor) -> None:
        if self.mask is None:
            self.slice(target)[:] = value
        else:
            self.slice(target)[self.mask] = value

    def _slice_attr_tensor(
        self,
        attr: str,
        default: int | float,
        remove_ints: bool,
        device: torch.device | str,
    ):
        if remove_ints:
            return torch.tensor(
                [
                    (default if sl is None else getattr(sl, attr))
                    for sl in self.slices
                    if not isinstance(sl, int)
                ],
                dtype=torch.long,
                device=device,
            )
        return torch.tensor(
            [
                (
                    default
                    if not isinstance(sl, slice) or (att := getattr(sl, attr)) is None
                    else att
                )
                for sl in self.slices
            ],
            dtype=torch.long,
            device=device,
        )

    def slice_starts_tensor(
        self, remove_ints: bool, device: torch.device | str
    ) -> Tensor:
        return self._slice_attr_tensor("start", 0, remove_ints, device)

    def slice_stops_tensor(
        self, remove_ints: bool, device: torch.device | str
    ) -> Tensor:
        return self._slice_attr_tensor("stop", torch.inf, remove_ints, device)

    def slice_steps_tensor(
        self, remove_ints: bool, device: torch.device | str
    ) -> Tensor:
        return self._slice_attr_tensor("step", 1, remove_ints, device)

    @overload
    def slice_indices(
        self, outer_indices: Tensor, return_mask: Literal[False] = False
    ) -> Tensor: ...
    @overload
    def slice_indices(
        self, outer_indices: Tensor, return_mask: Literal[True] = True
    ) -> tuple[Tensor, Tensor]: ...
    def slice_indices(
        self, outer_indices: Tensor, return_mask: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        assert all(
            sl.step is None or sl.step == 1
            for sl in self.slices
            if isinstance(sl, slice)
        )
        adjustment = self.slice_starts_tensor(
            remove_ints=False, device=outer_indices.device
        ).unsqueeze(-1)[: outer_indices.shape[0]]
        indices = outer_indices - adjustment
        mask = torch.ones_like(outer_indices[0], dtype=torch.bool)
        for i, sl in enumerate(self.slices):
            if isinstance(sl, int) and i < outer_indices.shape[0]:
                mask &= outer_indices[i] == sl
        mask &= (indices >= 0).all(dim=0)
        mask &= (
            indices
            % self.slice_steps_tensor(
                remove_ints=False, device=outer_indices.device
            ).unsqueeze(-1)
            == 0
        ).all(dim=0)
        rmdim = [isinstance(sl, int) for sl in self.slices] + [False] * (
            outer_indices.shape[0] - len(self.slices)
        )
        indices = torch.stack(
            [
                index
                for rm, index in zip(
                    rmdim[: indices.shape[0]], indices.unbind(), strict=True
                )
                if not rm
            ]
        )
        mask &= (
            indices
            < torch.tensor(
                self._inner_shape()[: indices.shape[0]], device=indices.device
            ).unsqueeze(-1)
        ).all(dim=0)
        if return_mask:
            return indices, mask
        if not mask.all():
            raise ValueError("Some indices out of bounds")
        return indices

    def mask_indices(self, sliced_indices: Tensor) -> Tensor:
        if self.mask is None:
            return sliced_indices
        maskrange = torch.arange(
            assert_isint(self.mask.sum().item()), device=self.mask.device
        )
        n = torch.ones_like(self.mask, dtype=torch.long) * (-1)
        n[self.mask] = maskrange
        return torch.cat(
            [
                n[sliced_indices[: len(self.mask.shape)]],
                sliced_indices[len(self.mask.shape) :],
            ]
        )

    def invert_indices_slicing(self, sliced_indices: Tensor) -> Tensor:
        assert all(
            sl.step is None or sl.step == 1
            for sl in self.slices
            if isinstance(sl, slice)
        )
        adjustment = self.slice_starts_tensor(
            remove_ints=True, device=sliced_indices.device
        ).unsqueeze(-1)
        if adjustment.shape[0] < sliced_indices.shape[0]:
            adjustment = torch.cat(
                [
                    adjustment,
                    torch.zeros(
                        sliced_indices.shape[0] - adjustment.shape[0],
                        1,
                        dtype=torch.long,
                        device=adjustment.device,
                    ),
                ]
            )
        assert (
            adjustment
            == torch.tensor(
                [
                    (0 if sl is None else (sl.start or 0))
                    for sl in self.padded_slices
                    if not isinstance(sl, int)
                ],
                device=sliced_indices.device,
                dtype=torch.long,
            ).unsqueeze(-1)
        ).all()
        indices = sliced_indices + adjustment
        if any(isinstance(sl, int) for sl in self.slices):
            for i, sl in enumerate(self.slices):
                if isinstance(sl, int):
                    indices = torch.cat(
                        [
                            indices[:i],
                            torch.tensor(
                                [sl], dtype=torch.long, device=indices.device
                            ).expand(1, indices.shape[1]),
                            indices[i:],
                        ]
                    )
        return indices

    @property
    def padded_slices(self) -> tuple[slice | int | None, ...]:
        return tuple(self.slices) + (None,) * (self.ndim_filtered() - len(self.slices))

    def invert_mask(self, inner_indices: Tensor) -> Tensor:
        if self.mask is None:
            return inner_indices
        return torch.cat(
            [
                self.mask.nonzero()[inner_indices[0]].transpose(-2, -1),
                inner_indices[1:],
            ],
            dim=0,
        )

    def invert_filter(self, inner_indices):
        return self.invert_indices_slicing(self.invert_mask(inner_indices))

    def to(self, *args, **kwargs) -> "Filter":
        return Filter(
            slices=self.slices,
            mask=self.mask.to(*args, **kwargs) if self.mask is not None else None,
            shape=self.virtual_shape,
        )

    @property
    def dimmap_i2o(self):
        return [i for i, sl in enumerate(self.slices) if isinstance(sl, slice)]

    @property
    def dimmap_o2i(self):
        return {i: o for o, i in enumerate(self.dimmap_i2o)}

    def slice(self, other: Tensor | DictBatch, ndim: int | None = None):
        if ndim is None:
            return other[self.slicing_tuple]
        return other[self.slicing_tuple[:ndim]]

    # def reduce(self, dim):
    #     slices = [sl for i, sl in enumerate(self.slices) if i != dim]
    #     shape = self.shape[:dim] + self.shape[dim + 1 :]
    #     # TODO check if mask needs reduction and assert shape=1 there
    #     # add and-reduce and or-reduce for shape != 1
    #     return Filter(slices=slices, mask=self._mask, shape=shape)


def index_sparse_with_bool(value: Tensor, mask: Tensor):
    """
    performs value[mask] for sparse value
    """
    assert value.shape[: mask.ndim] == mask.shape
    new_shape = [assert_isint(mask.count_nonzero().item())] + list(
        value.shape[mask.ndim :]
    )
    include = mask[value.indices()[: mask.ndim].split(1)].squeeze()
    z = torch.zeros_like(mask, dtype=torch.long, device=value.device)
    z[mask] = torch.arange(
        assert_isint(new_shape[0]), device=value.device, dtype=torch.long
    )
    new_indices = value.indices()[:, include]
    new_indices = torch.cat(
        [z[new_indices[: mask.ndim].split(1)], new_indices[mask.ndim :]], dim=0
    )
    new_values = value.values()[include]
    return torch.sparse_coo_tensor(
        indices=new_indices, values=new_values, size=new_shape
    )


@define
class FilteredTensor:
    """
    Represents a virtual larger tensor via a prefix mask/filter and a value
    tensor (or DictBatch).

    Conceptually, for some large tensor X:

        FilteredTensor(value=filter.apply(X), filter=filter)

    so that:

        self.filter.apply(X) == self.value

    When `value` is a DictBatch, we only support filtering along the
    batch (0th) dimension.
    """

    value: Tensor | DictBatch = field(
        init=True,
        validator=validators.instance_of((Tensor, DictBatch)),
    )
    filter: Filter = field(
        init=True,
        repr=False,
        validator=validators.instance_of(Filter),
    )

    @convert(value)
    @staticmethod
    def value_converter(value: Tensor | DictBatch) -> Tensor | DictBatch:
        if isinstance(value, Tensor) and value.is_sparse:
            return value.coalesce()
        return value

    def __attrs_post_init__(self):
        if self.filter.slicing_overhang_ndim() > 0:
            raise ValueError(
                f"Slicing overhang {self.filter.slicing_overhang_ndim()} is > 0,"
                " needs to be adderessed to support (and is not viable for DictBatch)"
            )

        # DictBatch branch: only enforce batch-dim invariants
        if isinstance(self.value, DictBatch):
            # Only batch dim filtering is supported
            if len(self.filter.slices) > 1:
                raise ValueError(
                    "For DictBatch, Filter.slices may slice at most the batch dimension"
                )

            # Value is expected to be already masked: len(value) == mask.sum()
            if self.filter.mask is not None:
                expected = assert_isint(self.filter.mask.sum().item())
                if len(self.value) != expected:
                    raise ValueError(
                        "DictBatch value is expected to be already masked by Filter.mask: "
                        f"len(value)={len(self.value)}, mask.sum()={expected}"
                    )
            return

        inner = self.filter._inner_shape()
        if self.filter.mask is not None:
            assert self.value.shape[0] == assert_isint(self.filter.mask.sum().item()), (
                f"Value shape at dimension 0 ({self.value.shape}) does not "
                f"match number of mask elements ({self.filter.mask.sum()})"
            )
        assert tuple(self.value.shape)[: self.filter.ndim_filtered()] == inner, (
            f"Value shape {tuple(self.value.shape)} does not match "
            f"filter inner shape {inner}"
        )

    @classmethod
    def from_value_and_mask(
        cls,
        value: Tensor | DictBatch,
        mask_obj: Tensor | NamedFilter | None,
    ) -> "FilteredTensor":
        """
        Construct from a *value* that already corresponds to `mask_obj`,
        plus the mask itself.

        Tensor case:
            - mask_obj.shape is the prefix shape of the virtual tensor
            - value.shape == (mask_obj.sum(), *rest)
        DictBatch case:
            - mask_obj is 1D over the batch dimension
            - each field tensor has shape (mask_obj.sum(), ...)
        """
        if isinstance(mask_obj, NamedFilter):
            mask_obj = mask_obj.filter

        # Infer device from value
        if isinstance(value, Tensor):
            device = value.device
        else:
            try:
                first = next(iter(value.values()))
            except StopIteration:
                raise ValueError("Cannot infer device from empty DictBatch") from None
            device = first.device

        if mask_obj is None:
            # "Unmasked" → full mask on the batch dimension / first dim
            if isinstance(value, Tensor):
                mask = torch.ones(value.shape[0], dtype=torch.bool, device=device)
                slices = [None] * value.ndim
                shape = tuple(value.shape)
            else:
                batch_size = value.batch_size
                mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                slices = [None]
                shape = (batch_size,)
            return cls(
                value=value,
                filter=Filter.construct_from_full_shape(
                    slices=slices, mask=mask, shape=shape
                ),
            )

        # mask provided
        if not isinstance(mask_obj, Tensor):
            raise TypeError("mask_obj must be a Tensor, NamedFilter or None")

        mask = mask_obj.to(device)

        if isinstance(value, Tensor):
            shape: tuple[int, ...] = tuple(mask.shape) + tuple(value.shape[1:])
            slices = [None] * len(shape)
        else:
            # DictBatch: only batch dim is modelled in shape
            if mask.ndim != 1:
                raise ValueError(
                    "For DictBatch, from_value_and_mask expects a 1D mask over "
                    "the batch dimension."
                )
            shape = tuple(mask.shape)  # e.g. (N,)
            slices = [None] * mask.ndim

        return cls(
            value=value,
            filter=Filter.construct_from_full_shape(
                slices=slices, mask=mask, shape=shape
            ),
        )

    @classmethod
    def from_unmasked_value(
        cls,
        value: Tensor | DictBatch,
        filter_obj: Filter | NamedFilter | Tensor | None,
        presliced: bool = False,
    ) -> "FilteredTensor":
        """
        Construct from an "unmasked" value and a filter that describes how
        to obtain it from the virtual large tensor.

        - If `filter_obj` is a tensor, it is interpreted as a boolean mask.
        - If `presliced` is True, `value` is assumed to already have `slices`
          applied, so we only apply the mask.
        """
        if filter_obj is None:
            return cls.from_value_and_mask(value, None)

        if isinstance(filter_obj, NamedFilter):
            filter_obj = filter_obj.filter

        if isinstance(filter_obj, Tensor):
            mask = filter_obj
            if isinstance(value, Tensor):
                shape: tuple[int, ...] = tuple(mask.shape) + tuple(value.shape[1:])
                slices = [None] * len(shape)
            else:
                # DictBatch: mask over batch dimension only
                if mask.ndim != 1:
                    raise ValueError(
                        "When value is DictBatch, mask tensor must be 1D "
                        "(over the batch dimension)."
                    )
                shape = tuple(mask.shape)
                slices = (
                    [None] * mask.ndim
                )  # TODO make sure post init checks for [1:] == [None,...] rather than len=1 (and for not [0] is int)
            filter_obj = Filter(slices=slices, mask=mask, shape=shape)

        if not isinstance(filter_obj, Filter):
            raise TypeError("filter_obj must be a Filter, Tensor, NamedFilter or None")

        if presliced:
            return cls(
                value=filter_obj.apply_mask(value),
                filter=filter_obj,
            )
        return cls(
            value=filter_obj.apply(value),
            filter=filter_obj,
        )

    # ------------------------------------------------------------------ #
    #                               Masking                              #
    # ------------------------------------------------------------------ #

    @overload
    def mask_by_other(
        self,
        other: "Filter | Tensor | FilteredTensor",
        return_ft: Literal[True],
        presliced: bool | None,
        value_like: bool = False,
    ) -> "FilteredTensor": ...

    @overload
    def mask_by_other(
        self,
        other: "Filter | Tensor | FilteredTensor",
        return_ft: Literal[False],
        presliced: bool | None,
        value_like: bool = False,
    ) -> Tensor | DictBatch: ...

    def mask_by_other(
        self,
        other: "Filter | Tensor | FilteredTensor",
        return_ft: bool = False,
        presliced: bool | None = None,
        value_like: bool = False,
    ) -> "Tensor | DictBatch | FilteredTensor":
        """
        Apply an additional mask `other` on top of this FilteredTensor.

        - If `other` is a Filter, we require it to describe the same
          virtual tensor (same `shape` and `slices`).
        - If `other` is a tensor mask, `presliced` controls whether that
          mask already matches the filter's sliced view.
        - If `value_like` is True, the mask is interpreted in the
          "value space" (shape matching self.value), otherwise in the
          virtual large tensor space.
        """
        if isinstance(other, FilteredTensor):
            other = other.filter

        mask: Tensor | None = None

        if isinstance(other, Filter):
            if self.filter.shape != other.shape or (
                self.filter.slicing_tuple != other.slicing_tuple
            ):
                raise ValueError(
                    "Filters must describe the same logical view to be combined"
                )
            assert presliced is None
            mask = other.mask
            if mask is None:
                return self
        elif isinstance(other, Tensor):
            mask = other
            if not presliced:
                mask = self.filter.slice(mask)
        else:
            if presliced is not None:
                raise ValueError(
                    "presliced has no effect when other is not Tensor or Filter"
                )
            assert False

        assert mask is not None
        selfmask = self.filter.mask
        # if selfmask is None:
        #     raise ValueError("Self filter must have a mask to combine")

        # Map global/local mask into value space
        if value_like or selfmask is None:
            # mask already lives in the local (value) space
            filtered_mask = mask
        else:
            # mask is in the global space; restrict to currently-selected docs
            if selfmask.ndim > mask.ndim:
                mask = right_expand(mask, selfmask.shape)
            filtered_mask = mask[selfmask]

        # Apply to value -------------------------------------------------
        if isinstance(self.value, DictBatch):
            if filtered_mask.ndim != 1:
                raise ValueError(
                    "For DictBatch, combined mask must be 1D over the batch dim"
                )
            new_value = self.value[filtered_mask]
        else:
            if self.value.is_sparse:
                v = self.value
                assert isinstance(v, Tensor)
                if not v.is_coalesced():
                    v = v.coalesce()
                new_value = index_sparse_with_bool(v, filtered_mask)
            else:
                new_value = self.value[filtered_mask]
        if not return_ft:
            return new_value

        # Build the new global mask

        if selfmask is None:
            outmask = filtered_mask
        else:
            if selfmask.ndim < mask.ndim:
                selfmask = right_expand(selfmask, mask.shape)
            if value_like:
                # scatter local mask back into global positions where selfmask is True
                outmask = torch.zeros_like(selfmask).masked_scatter(
                    selfmask, filtered_mask
                )
            else:
                outmask = selfmask & mask

        return FilteredTensor(
            value=new_value,
            filter=Filter(
                slices=self.filter.slices,
                mask=outmask,
                shape=self.filter.shape,
            ),
        )

    def filter_inactive_docs(self) -> "FilteredTensor":
        if isinstance(self.value, DictBatch):
            assert False
        if self.value.is_sparse:
            newmask = torch.zeros(
                (
                    assert_isint(self.filter.mask.sum().item())
                    if self.filter.mask is not None
                    else self.value.shape[0]
                ),
                dtype=torch.bool,
                device=self.value.device,
            )
            newmask[self.value.indices()[0]] = True
        else:
            newmask = self.value > 0
            while newmask.ndim > 1:
                newmask = newmask.any(dim=-1)
            assert (
                self.filter.mask is None
                or newmask.shape[0] == self.filter.mask.sum().item()
            )
        return self.mask_by_other(
            newmask, return_ft=True, presliced=True, value_like=True
        )

    def to_filtered_like_self(
        self,
        t: Tensor,
        presliced: bool = True,
        premasked: bool = True,
        ndim: int | None = None,
    ) -> "FilteredTensor":
        if premasked and not presliced:
            raise ValueError("Cannot be masked and not sliced")
        if not presliced:
            t = self.filter.slice(t, ndim=ndim)
        if not premasked:
            t = t[self.filter.mask]
        filt = self.filter
        if ndim is not None:
            filt = Filter(
                slices=self.filter.slices[:ndim],
                mask=self.filter.mask,
                shape=self.filter.shape[:ndim],
            )
        return FilteredTensor(value=t, filter=filt)

    # ------------------------------------------------------------------ #
    #                  Indexing / "torch-like" helpers                   #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"FilteredTensor(value={self.value}, mask={self.filter})"

    ### indexing
    def internalize_indices(self, indices: Tensor):
        assert indices.ndim == 2
        ids, mask = self.filter.slice_indices(indices, return_mask=True)
        ids = ids[:, mask]
        ids = self.filter.mask_indices(ids)
        new_mask = mask.clone()
        new_mask[mask] = ids >= 0
        ids = ids[:, (ids >= 0).all(dim=0)]
        return ids, new_mask

    def externalize_indices(self, indices: Tensor):
        if indices.ndim != 2:
            raise ValueError("Indices must be 2D")
        return self.filter.invert_filter(indices)

    def index_where_valid(self, indices: Tensor):
        ids, mask = self.internalize_indices(indices)
        return self.value[ids.unbind()], mask

    def __getitem__(self, i):
        if isinstance(i, Tensor) and i.dtype == torch.long and i.ndim == 2:
            si = self.filter.slice_indices(i)
            indices = self.filter.mask_indices(si)
            if (indices == -1).any():
                raise IndexError("Some indexed values are excluded by the mask")
            if self.is_sparse:
                if i.shape[0] != 1:
                    raise ValueError(
                        "Sparse tensors can only be indexed with 1D tensors"
                    )
                return self.value.index_select(0, indices)
            return self.value[indices.unbind()]

        raise NotImplementedError(
            "Indexing FilteredTensor is only implemented for 2D integer tensors"
        )

    def index_select(self, index: Tensor, dim: int):
        if dim != 0:
            raise NotImplementedError("Only dim=0 is implemented")
        assert index.ndim == 1
        index = index.unsqueeze(0)
        si = self.filter.slice_indices(index)
        inner_index = self.filter.mask_indices(si)
        if (inner_index == -1).any():
            raise IndexError("Some indexed values are excluded by the filter")

        inner_index = inner_index.squeeze(0)
        if isinstance(self.value, DictBatch):
            return self.value[inner_index]
        return self.value.index_select(dim, inner_index)

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

    def to_dense_unfiltered(self, default_value: int | float = 0) -> Tensor:
        # TODO remove?
        """
        Materialise the full virtual tensor as a dense tensor.

        Only defined when the underlying value is a tensor.
        """
        if isinstance(self.value, DictBatch):
            raise TypeError("to_dense_unfiltered is only defined for tensor values")

        z = torch.full(
            self.shape,
            default_value,
            dtype=self.value.dtype,
            device=self.value.device,
        )
        dense_value = self.value.to_dense() if self.is_sparse else self.value
        self.filter.slice(z)[self.filter.mask] = dense_value
        return z

    def to_sparse(self) -> "FilteredTensor":
        if isinstance(self.value, DictBatch):
            raise TypeError("to_sparse is only defined for tensor values")
        if self.is_sparse:
            return self
        return FilteredTensor(
            value=torch.sparse_coo_tensor(
                indices=self.value.indices(),
                values=self.value.values(),
                size=self.value.shape,
            ),
            filter=self.filter,
        )

    def to_dense(self) -> "FilteredTensor":
        if isinstance(self.value, DictBatch):
            raise TypeError("to_dense is only defined for tensor values")
        if not self.is_sparse:
            return self
        return FilteredTensor(value=self.value.to_dense(), filter=self.filter)

    def indices(self) -> Tensor:
        if not self.is_sparse or isinstance(self.value, DictBatch):
            raise TypeError("indices() is only defined for sparse tensor values")
        return self.filter.invert_filter(self.value.indices())  # type: ignore[arg-type]

    def values(self) -> Tensor:
        if not self.is_sparse or isinstance(self.value, DictBatch):
            raise TypeError("values() is only defined for sparse tensor values")
        return self.value.values()  # type: ignore[call-arg]

    def nonzero(self) -> Tensor:
        if isinstance(self.value, DictBatch):
            raise TypeError("nonzero() is only defined for tensor values")
        return self.filter.invert_filter(
            self.value.nonzero().transpose(-2, -1)  # type: ignore[call-arg]
        )

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.value, Tensor) and self.value.is_sparse

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the virtual large tensor.

        For DictBatch this is just the shape tracked by the Filter (typically
        a 1‑tuple `(batch_size,)`).
        """
        if isinstance(self.value, DictBatch):
            return DictBatchShapes(
                batch_sizes=self.filter.virtual_shape,
                shapes=self.value.shapes,
            )
        return self.filter.virtual_shape + self.value.shape[1:]

    def cuda(self) -> "FilteredTensor":
        return self.to(torch.device("cuda"))

    def to(self, *args, **kwargs) -> "FilteredTensor":
        filter_kwargs = kwargs
        if "dtype" in kwargs:
            filter_kwargs = {**kwargs, "dtype": torch.bool}
        filter_args = args
        for i, arg in enumerate(args):
            if isinstance(arg, torch.dtype):
                filter_args = filter_args[:i] + (filter_kwargs,) + filter_args[i + 1 :]
        return FilteredTensor(
            value=self.value.to(*args, **kwargs),
            filter=self.filter.to(*filter_args, **filter_kwargs),
        )

    # def select(self, select_fn):
    #     if self.value.is_sparse:
    #         self.value = self.value.coalesce()
    #         vmask: Tensor = select_fn(self.value.values())
    #         nv = torch.sparse_coo_tensor(
    #             indices=self.value.indices()[:, vmask],
    #             values=self.value.values()[vmask],
    #             size=[vmask.count_nonzero(), *self.value.shape[1:]],
    #         )
    #         mask = torch.zeros_like(self.filter.mask)
    #         mask[self.filter.mask] = vmask
    #         return FilteredTensor(
    #             value=nv,
    #             filter=Filter(
    #                 slices=self.filter.slices, mask=mask, shape=self.filter.shape
    #             ),
    #         )
    #     vmask = select_fn(self.value)
    #     mask = torch.zeros_like(self.filter.mask)
    #     mask[self.filter.mask] = vmask
    #     return FilteredTensor(
    #         value=self.value[vmask],
    #         filter=Filter(
    #             slices=self.filter.slices, mask=mask, shape=self.filter.shape
    #         ),
    #     )


def checker1(shape, slice_dims, unmasked_dims=0):
    fts = []
    numel = torch.prod(torch.tensor(shape)).item()
    big_arange = torch.arange(numel).reshape(shape)

    def slice4(dimshape):
        sl12a = slice(dimshape // 2, None, 2)
        sl12b = slice(dimshape // 2 + 1, None, 2)
        sl22a = slice(None, dimshape // 2, 2)
        sl22b = slice(1, dimshape // 2, 2)
        return sl12a, sl12b, sl22a, sl22b

    slices = [[i] for i in slice4(shape[0])]
    for i in range(1, slice_dims):
        prev_slices = slices
        slices = []
        for sl in prev_slices:
            for sl2 in slice4(shape[i]):
                slices.append(sl + [sl2])
    fts = []
    for sl_l in slices:
        mshape = [sh // 4 for sh in shape[:slice_dims]] + shape[slice_dims:]
        if unmasked_dims:
            mshape = mshape[:-unmasked_dims]
        mask_a = torch.rand(*mshape) > 0.5
        mask_b = ~mask_a
        fts.append(
            FilteredTensor.from_unmasked_value(
                big_arange, Filter(sl_l, mask_a, shape=shape)
            )
        )
        fts.append(
            FilteredTensor.from_unmasked_value(
                big_arange, Filter(sl_l, mask_b, shape=shape)
            )
        )

    z = torch.zeros(shape, dtype=torch.long)

    for ft in fts:
        assert (ft.filter.apply(z) == 0).all()
        ft.filter.slice(z)[ft.filter.mask] = ft.value
        assert (ft.filter.apply(z) == ft.value).all()
    assert (z == big_arange).all()
    print("success")


# def main():
#     for i in range(1, 4):
#         checker1([128, 128, 128, 128], i)

#     value = torch.arange(20)
#     slice_mask = slice(2, 14, 2)
#     tensor_mask = torch.tensor([True, False, True, True, False, True])
#     mask = Filter(slices=[slice_mask], mask=tensor_mask, shape=(20,))

#     ft = FilteredTensor.from_unmasked_value(value, mask)

#     t = torch.arange(10, 30)
#     filtered_t = ft.filter.apply(t)
#     result = ft.value + filtered_t

#     ft_result = ft.to_filtered_like_self(result)

#     dense_result = ft_result.to_dense()
#     print(dense_result)


# if __name__ == "__main__":
#     main()

from saeco.evaluation.filtered_modular import Filter, FilteredTensor
