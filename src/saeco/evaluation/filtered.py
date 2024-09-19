from typing import List, Tuple, Union

import torch
from attr import define, field
from torch import Tensor


@define
class SliceMask:
    slices: Tuple[slice, ...]
    shape: Tuple[int, ...] = field(converter=tuple)

    def __post_init__(self):
        assert len(self.slices) == len(self.shape)
        for i, slc in enumerate(self.slices):
            assert isinstance(slc, slice)
            dim_size = self.shape[i]
            start = slc.start if slc.start is not None else 0
            stop = slc.stop if slc.stop is not None else dim_size
            assert (
                0 <= start <= dim_size
            ), f"Slice start {start} out of bounds for dimension {i}"
            assert (
                0 <= stop <= dim_size
            ), f"Slice stop {stop} out of bounds for dimension {i}"
            assert (
                start <= stop
            ), f"Slice start {start} greater than stop {stop} for dimension {i}"

    def numel(self) -> int:
        n = 1
        for nd in self.numels_per_dim():
            n *= nd
        return n

    def numels_per_dim(self) -> int:
        l = []
        for slc, dim in zip(self.slices, self.shape):
            step = slc.step or 1
            start = slc.start or 0
            stop = slc.stop if slc.stop is not None else dim
            length = max(0, (stop - start + (step - 1)) // step)
            l.append(length)
        return l

    def apply(self, tensor: Tensor) -> Tensor:
        if tensor.shape[: len(self.shape)] != self.shape:
            raise ValueError(
                f"Shape mismatch: tensor shape {tensor.shape} does not match mask shape {self.shape}"
            )
        return tensor[self.slices]

    def __repr__(self):
        return f"SliceMask(slices={self.slices}, shape={self.shape})"


@define
class Filt:
    slices: list[SliceMask]
    mask: Tensor | None

    @property
    def masks(self) -> List[Union[Tensor, SliceMask]]:
        return self.slices + ([self.mask] if self.mask is not None else [])

    def __attrs_post_init__(self):
        current_shape = self.masks[0].shape
        for i, mask in enumerate(self.slices[1:] + [self.mask], start=1):
            prev_output_shape = self._get_output_shape(
                self.slices[i - 1], current_shape
            )
            mask_shape = mask.shape
            if len(prev_output_shape) > len(mask_shape):
                raise ValueError("Mask shape is too small")
            if prev_output_shape != list(mask_shape[: len(prev_output_shape)]):
                raise ValueError(
                    f"Mask shapes are not compatible when sequentially applied: "
                    f"expected {prev_output_shape}, got {mask_shape} at position {i}"
                )
            current_shape = mask_shape

    def _get_output_shape(self, mask, input_shape) -> Tuple[int, ...]:
        if isinstance(mask, Tensor):
            num_true = int(mask.sum().item())
            return (num_true,) + input_shape[1:]
        elif isinstance(mask, SliceMask):
            return mask.numels_per_dim() + list(input_shape[len(mask.shape) :])
        else:
            raise TypeError(f"Unsupported mask type {type(mask)}")

    @property
    def shape(self) -> Tuple[int, ...]:
        shape = list(self.masks[0].shape)
        for mask in self.masks[1:]:
            if len(mask.shape) > len(shape):
                shape = shape + list(mask.shape[len(shape) :])
        return shape

    def apply(self, tensor: Tensor) -> Tensor:
        t = tensor
        for mask in self.masks:
            if isinstance(mask, SliceMask):
                t = mask.apply(t)
            else:
                t = t[mask]
        return t

    def intersect(self, other: "Filt") -> "Filt":
        raise NotImplementedError(
            "Intersect not implemented, need to make slice intersect"
        )

    def slice(self, other):
        t_view = other
        for mask in self.slices:
            t_view = mask.apply(t_view)
        return t_view

    def writeat(self, target: Tensor, value):
        self.slice(target)[self.mask] = value


def mask_to_tensor(mask: Union[Tensor, SliceMask]) -> Tensor:
    if isinstance(mask, Tensor):
        return mask
    elif isinstance(mask, SliceMask):
        tensor_mask = torch.zeros(mask.shape, dtype=torch.bool)
        tensor_mask[mask.slices] = True
        return tensor_mask
    else:
        raise TypeError(f"Unsupported mask type {type(mask)}")


def index_sparse_with_bool(value, mask):
    new_shape = [mask.count_nonzero()] + list(value.shape[mask.ndim :])
    include = mask[value.indices()[: mask.ndim].split(1)].squeeze()
    z = torch.zeros_like(mask, dtype=torch.long)
    z[mask] = torch.arange(new_shape[0], device=value.device, dtype=torch.long)
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
    value: Tensor
    filter: Filt

    def __attrs_post_init__(self):
        assert (
            self.value.shape[0] == self.filter.mask.sum()
        ), f"Value shape at dimension 0 ({self.value.shape}) does not match number of mask elements ({self.filter.mask.sum()})"

    @classmethod
    def from_unmasked_value(cls, value: Tensor, filter: Filt):
        value_filtered = filter.apply(value)
        return cls(value=value_filtered, filter=filter)

    def mask_by_other(self, other_mask: Filt) -> Tensor:
        assert self.filter.slices == other_mask.slices
        mask = other_mask.mask
        if self.filter.mask.ndim > mask.ndim:
            mask = mask.expand(self.filter.mask.shape)
        if self.value.is_sparse:
            if not self.value.is_coalesced():
                self.value = self.value.coalesce()
            return index_sparse_with_bool(self.value, mask[self.filter])
        return self.value[mask[self.filter]]

    def to_dense(self, default_value=0) -> Tensor:
        z = torch.full(self.filter.shape, default_value, dtype=self.value.dtype)
        self.filter.slice(z)[self.filter.mask] = self.value
        return z

    def to_filtered_like_self(self, t: Tensor) -> "FilteredTensor":
        # t_filtered = self.filter.apply(t)
        return FilteredTensor(value=t, filter=self.filter)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.filter.shape + self.value.shape[1:]

    def cuda(self) -> "FilteredTensor":
        return self.to(torch.device("cuda"))

    def to(self, *args, **kwargs) -> "FilteredTensor":
        return FilteredTensor(
            value=self.value.to(*args, **kwargs),
            filter=Filt(
                slices=list(self.filter.slices),
                mask=self.filter.mask.to(*args, **kwargs),
            ),
        )

    def __repr__(self):
        return f"FilteredTensor(value={self.value}, mask={self.filter})"


def checker1(shape, slice_dims, a, b, c, unmasked_dims=0):
    fts = []
    numel = torch.prod(torch.tensor(shape)).item()
    big_arange = torch.arange(numel).reshape(shape)

    # sl1 = slice(shape[0] // 2, None)
    # sl2 = slice(None, shape[0] // 2)
    def slice4(dimshape):
        sl12a = slice(dimshape // 2, None, 2)
        sl12b = slice(dimshape // 2 + 1, None, 2)
        sl22a = slice(None, dimshape // 2, 2)
        sl22b = slice(1, dimshape // 2, 2)
        return sl12a, sl12b, sl22a, sl22b

    def slm4(dimshape):
        return [SliceMask(sl, (dimshape,)) for sl in slice4(dimshape)]

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
                big_arange, Filt([SliceMask(sl_l, shape=shape[:slice_dims])], mask_a)
            )
        )
        fts.append(
            FilteredTensor.from_unmasked_value(
                big_arange, Filt([SliceMask(sl_l, shape=shape[:slice_dims])], mask_b)
            )
        )

    z = torch.zeros(shape, dtype=torch.long)

    for ft in fts:
        assert (ft.filter.apply(z) == 0).all()
        ft.filter.slice(z)[ft.filter.mask] = ft.value
        assert (ft.filter.apply(z) == ft.value).all()
    assert (z == big_arange).all()
    print("success")


def checker2(shape, slice_dims, a, b, c, unmasked_dims=0):
    fts = []
    numel = torch.prod(torch.tensor(shape)).item()
    big_arange = torch.arange(numel).reshape(shape)

    # sl1 = slice(shape[0] // 2, None)
    # sl2 = slice(None, shape[0] // 2)
    def slice4(dimshape):
        d = dimshape // 2
        sl12a = slice(d, d * 2, 2)
        sl12b = slice(d + 1, d * 2, 2)
        sl22a = slice(0, d, 2)
        sl22b = slice(1, d, 2)
        return sl12a, sl12b, sl22a, sl22b

    def slm4(dimshape):
        return [SliceMask(sl, (dimshape,)) for sl in slice4(dimshape)]

    slices = [[i] for i in slice4(shape[0])]
    for i in range(1, slice_dims):
        prev_slices = slices
        slices = []
        for sl in prev_slices:
            for sl2 in slice4(shape[0] // (4**i)):
                slices.append(sl + [sl2])
    fts = []
    for sl_l in slices:
        mshape = [sh // (4**slice_dims) for sh in shape[:1]] + shape[1:]
        if unmasked_dims:
            mshape = mshape[:-unmasked_dims]
        mask_a = torch.rand(*mshape) > 0.5
        mask_b = ~mask_a
        fts.append(
            FilteredTensor.from_unmasked_value(
                big_arange,
                Filt(
                    [
                        SliceMask([sl], shape=[shape[0] // (4**i)])
                        for i, sl in enumerate(sl_l)
                    ],
                    mask_a,
                ),
            )
        )
        fts.append(
            FilteredTensor.from_unmasked_value(
                big_arange,
                Filt(
                    [
                        SliceMask([sl], shape=[shape[0] // (4**i)])
                        for i, sl in enumerate(sl_l)
                    ],
                    mask_b,
                ),
            )
        )

    z = torch.zeros(shape, dtype=torch.long)

    for ft in fts:
        assert (ft.filter.apply(z) == 0).all()
        ft.filter.slice(z)[ft.filter.mask] = ft.value
        assert (ft.filter.apply(z) == ft.value).all()
    assert (z == big_arange).all()
    print("success")


def main():

    checker1([128, 128, 128, 128], 2, 1, 2, 3)

    checker2([128, 128, 128, 128], 2, 1, 2, 3)

    value = torch.arange(20)
    slice_mask = SliceMask(slices=(slice(2, 14, 2),), shape=(20,))
    tensor_mask = torch.tensor([True, False, True, True, False, True])
    mask = Filt(slices=[slice_mask], mask=tensor_mask)

    ft = FilteredTensor.from_unmasked_value(value, mask)

    t = torch.arange(10, 30)
    filtered_t = ft.filter.apply(t)
    result = ft.value + filtered_t

    ft_result = ft.to_filtered_like_self(result)

    dense_result = ft_result.to_dense()
    print(dense_result)


if __name__ == "__main__":
    main()
