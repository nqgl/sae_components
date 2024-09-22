from typing import List, Tuple, Union

import attr

import torch
from attrs import Converter, define, field
from torch import Tensor


def convert(fld, takes_self=False, takes_field=False):
    def converter_wrapper(fn):
        assert fld.converter is None
        fld.converter = Converter(fn, takes_self=takes_self, takes_field=takes_field)
        return fn

    return converter_wrapper


@define
class SliceMask:
    slices: Tuple[slice, ...]
    shape: Tuple[int, ...] = field()

    @convert(shape)
    @staticmethod
    def shape_converter(value):
        return tuple(value)

    @classmethod
    def from_range(self, _from, _to, shape):
        return SliceMask(slices=(slice(_from, _to),), shape=shape)

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
class Filter:
    slices: list[SliceMask] = field()
    _mask: Tensor | torch.device = field()

    @convert(slices)
    @staticmethod
    def slices_converter(value):
        if isinstance(value, SliceMask):
            return [value]
        return value

    @property
    def mask(self):
        if isinstance(self._mask, Tensor):
            return self._mask
        raise ValueError("Mask not available")

    @property
    def masks(self) -> List[Union[Tensor, SliceMask]]:
        return self.slices + ([self._mask] if isinstance(self._mask, Tensor) else [])

    def __attrs_post_init__(self):
        current_shape = self.masks[0].shape
        for i, mask in enumerate(self.masks[1:], start=1):
            prev_output_shape = self._get_output_shape(
                self.slices[i - 1], current_shape
            )
            mask_shape = mask.shape
            if len(prev_output_shape) > len(mask_shape):
                raise ValueError("Mask shape is too small")
            if not all(
                [
                    a == b or b == 1
                    for a, b in zip(
                        prev_output_shape, list(mask_shape[: len(prev_output_shape)])
                    )
                ]
            ):
                raise ValueError(
                    f"Mask shapes are not compatible when sequentially applied: "
                    f"expected {prev_output_shape}, got {mask_shape} at position {i}"
                )
            current_shape = mask_shape
        if not isinstance(self._mask, Tensor):
            mask = torch.ones(1, dtype=torch.bool, device=self._mask or "cpu")
            # this is the wrong place to be doing this
            # the mask needs to be a size that the ft knows but F does not know
            # solve by:
            # 1.  refactor to use factories for construction everywhere and make mask from none here
            # 2.  fully explicitly support None for Mask and fix all the annoying edge cases there
            # hmm wait maybe its being weird with features because the features are along the last dimension?
            # self._mask = mask
            expected = self._get_output_shape(self.slices[-1], current_shape)
            self._mask = mask.expand(expected[0], *([1] * len(expected[1:])))

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
        self.slice(tensor)
        return self.slice(tensor)[self.mask]

    def intersect(self, other: "Filter") -> "Filter":
        raise NotImplementedError(
            "Intersect not implemented, need to make slice intersect"
        )

    def slice(self, other):
        t_view = other
        for mask in self.slices:
            t_view = mask.apply(t_view)
        return t_view

    def apply_mask(self, tensor: Tensor) -> Tensor:
        if self._mask is None:
            return tensor
        if tensor.is_sparse:
            if not tensor.is_coalesced():
                tensor = tensor.coalesce()

            return index_sparse_with_bool(tensor, self.mask)
        return tensor[self.mask]

    def writeat(self, target: Tensor, value):
        self.slice(target)[self.mask] = value

    @classmethod
    def from_whatever(cls, value):
        if isinstance(value, Filter):
            return value
        if isinstance(value, SliceMask):
            return cls([value], None)
        if isinstance(value, Tensor):
            return cls([], value)
        if isinstance(value, tuple | list):
            if all(isinstance(v, SliceMask) for v in value):
                return cls(value, None)
            if all(isinstance(v, slice) for v in value):
                raise NotImplementedError("")
                # return cls([SliceMask(slices=value)], None)
            elif len(value) == 2:
                slices, mask = value
                assert isinstance(mask, Tensor | None)
                if isinstance(slices, list):
                    return cls(slices, mask)
                elif isinstance(slices, SliceMask):
                    return cls([slices], mask)
                else:
                    raise TypeError(f"Unsupported type {type(slices)}")
        else:
            raise ValueError(
                f"Unsupported type(s) for automatic conversion to Filter {value}"
            )


def slice_to_tensor(mask: Union[Tensor, SliceMask]) -> Tensor:
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
    filter: Filter = field(init=True, repr=False)

    @convert(filter, takes_self=True)
    @staticmethod
    def filter_converter(filt, inst):
        return Filter.from_whatever(filt)

    def __attrs_post_init__(self):
        if self.filter._mask is not None:
            assert (
                self.value.shape[0] == self.filter.mask.sum()
            ), f"Value shape at dimension 0 ({self.value.shape}) does not match number of mask elements ({self.filter.mask.sum()})"
        else:
            assert True  # TODO
            # self.value.shape[0] == self.filter

    @classmethod
    def from_unmasked_value(cls, value: Tensor, filter: Filter, presliced=False):
        if presliced:
            # if value.is_sparse:
            # if not value.is_coalesced():
            #     value = value.coalesce()
            # return cls(
            #     value=index_sparse_with_bool(value=value, mask=filter.mask),
            #     filter=filter,
            # )
            return cls(value=filter.apply_mask(value), filter=filter)
        return cls(value=filter.apply(value), filter=filter)

    def mask_by_other(
        self,
        other: Union[Filter, Tensor, "FilteredTensor"],
        return_ft=False,
        presliced=None,
    ) -> Union[Tensor, "FilteredTensor"]:
        if isinstance(other, FilteredTensor):
            other = other.filter
        if isinstance(other, Filter):
            assert self.filter.slices == other.slices
            mask = other.mask
        if isinstance(other, Tensor):
            mask = other
            if not presliced:
                mask = self.filter.slice(mask)
        else:
            if presliced is not None:
                raise ValueError("presliced has no effect when other is not Tensor")
        if mask == slice(None, None):
            filtered_mask = torch.ones(  # todo handle self.filter._mask == None
                self.filter.mask.sum(), dtype=torch.bool, device=self.value.device
            )
        else:
            if self.filter.mask.ndim > mask.ndim:
                mask = right_expand(mask, self.filter.mask.shape)
            filtered_mask = mask[self.filter.mask]

        if self.value.is_sparse:
            if not self.value.is_coalesced():
                self.value = self.value.coalesce()
            value = index_sparse_with_bool(self.value, filtered_mask)
        else:
            value = self.value[filtered_mask]
        if not return_ft:
            return value
        selfmask = self.filter.mask
        if selfmask.ndim < mask.ndim:
            selfmask = right_expand(selfmask, mask.shape)
        return FilteredTensor(
            value=value,
            filter=Filter(
                slices=self.filter.slices,
                mask=selfmask & mask,
            ),
        )

    def to_dense(self) -> "FilteredTensor":
        return FilteredTensor(value=self.value.to_dense(), filter=self.filter)

    def to_dense_unfiltered(self, default_value=0) -> Tensor:
        z = torch.full(self.shape, default_value, dtype=self.value.dtype)
        self.filter.slice(z)[self.filter.mask] = self.value.to_dense()
        return z

    def to_filtered_like_self(self, t: Tensor) -> "FilteredTensor":
        # t_filtered = self.filter.apply(t)
        return FilteredTensor(value=t, filter=self.filter)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.filter.shape + list(self.value.shape[1:])

    def cuda(self) -> "FilteredTensor":
        return self.to(torch.device("cuda"))

    def to(self, *args, **kwargs) -> "FilteredTensor":
        return FilteredTensor(
            value=self.value.to(*args, **kwargs),
            filter=Filter(
                slices=list(self.filter.slices),
                mask=(
                    self.filter.mask.to(*args, **kwargs)
                    if self.filter._mask is not None
                    else self.filter.mask
                ),
            ),
        )

    def __repr__(self):
        return f"FilteredTensor(value={self.value}, mask={self.filter})"

    def __getitem__(self, i):
        if isinstance(i, int) or isinstance(i, Tensor):
            # super inefficient just proof of concept
            for slc in self.filter.slices:
                slc = slc.slices[0]
                i *= slc.step or 1
                i += slc.start or 0
                if slc.stop is not None:
                    if i >= slc.stop:
                        raise IndexError(f"Index {i} out of bounds for slice {slc}")
            return self.value[
                self.filter.mask.nonzero()[i]
            ]  # something like this but probably not this
        raise NotImplementedError("todo if needed")


def right_expand(t, shape):
    assert len(shape) >= len(t.shape)
    assert all(
        [
            t.shape[i] == shape[i] or shape[i] == -1 or t.shape[i] == 1
            for i in range(len(t.shape))
        ]
    ), f"Shapes do not match for right-expand: {t.shape} and {shape}"
    for i in range(len(shape) - len(t.shape)):
        t = t.unsqueeze(-1)
    return t.expand(shape)


def checker1(shape, slice_dims, unmasked_dims=0):
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
                big_arange, Filter([SliceMask(sl_l, shape=shape[:slice_dims])], mask_a)
            )
        )
        fts.append(
            FilteredTensor.from_unmasked_value(
                big_arange, Filter([SliceMask(sl_l, shape=shape[:slice_dims])], mask_b)
            )
        )

    z = torch.zeros(shape, dtype=torch.long)

    for ft in fts:
        assert (ft.filter.apply(z) == 0).all()
        ft.filter.slice(z)[ft.filter.mask] = ft.value
        assert (ft.filter.apply(z) == ft.value).all()
    assert (z == big_arange).all()
    print("success")


def checker2(shape, slice_dims, unmasked_dims=0):
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
                Filter(
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
                Filter(
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
    for i in range(1, 4):
        checker1([128, 128, 128, 128], i)
        checker2([128, 128, 128, 128], i)

    value = torch.arange(20)
    slice_mask = SliceMask(slices=(slice(2, 14, 2),), shape=(20,))
    tensor_mask = torch.tensor([True, False, True, True, False, True])
    mask = Filter(slices=[slice_mask], mask=tensor_mask)

    ft = FilteredTensor.from_unmasked_value(value, mask)

    t = torch.arange(10, 30)
    filtered_t = ft.filter.apply(t)
    result = ft.value + filtered_t

    ft_result = ft.to_filtered_like_self(result)

    dense_result = ft_result.to_dense()
    print(dense_result)


if __name__ == "__main__":
    main()
