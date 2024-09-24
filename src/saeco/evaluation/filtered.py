from typing import List, Tuple, Union

import torch
from attrs import Converter, define, field, validators
from torch import Tensor


def convert(fld, takes_self=False, takes_field=False):
    def converter_wrapper(fn):
        assert fld.converter is None
        fld.converter = Converter(fn, takes_self=takes_self, takes_field=takes_field)
        return fn

    return converter_wrapper


def slice_shape(input_shape, slices: list[slice]) -> Tuple[int, ...]:
    if isinstance(slices, slice):
        slices = [slices]
    out_shape = []
    assert len(input_shape) >= len(slices)
    for slc, shape in zip(slices, input_shape):
        if isinstance(slc, int):
            continue
        if slc is None:
            out_shape.append(shape)
            continue
        slmin = slc.start or 0
        slmax = min(slc.stop or shape, shape)
        step = slc.step or 1
        out_shape.append((slmax - slmin + step - 1) // step)
    return tuple(out_shape + list(input_shape[len(slices) :]))


@define
class Filter:
    slices: list[slice | int] = field()
    _mask: Tensor | torch.device = field()
    shape: Tuple[int, ...] = field()

    @property
    def mask(self):
        if isinstance(self._mask, Tensor):
            return self._mask
        raise ValueError("Mask not available")

    def __attrs_post_init__(self):
        sliced = slice_shape(self.shape, self.slices)
        if not isinstance(self._mask, Tensor):
            mask = torch.ones(1, dtype=torch.bool, device=self._mask or "cpu")
            self._mask = mask.expand(sliced[0])
        assert tuple(self.mask.shape) == sliced[: len(self.mask.shape)]

    def _inner_shape(self):
        sliced = slice_shape(self.shape, self.slices)
        assert tuple(self.mask.shape) == sliced[: len(self.mask.shape)]
        return (self.mask.sum(),) + sliced[len(self.mask.shape) :]

    def apply(self, tensor: Tensor) -> Tensor:
        self.slice(tensor)
        return self.slice(tensor)[self.mask]

    def intersect(self, other: "Filter") -> "Filter":
        raise NotImplementedError(
            "Intersect not implemented, need to make slice intersect"
        )

    @property
    def slicing_tuple(self):
        return tuple([sl if sl is not None else slice(None) for sl in self.slices])

    def slice(self, other):
        return other[self.slicing_tuple]

    def apply_mask(self, tensor: Tensor) -> Tensor:
        if self._mask is None:
            return tensor
        if tensor.is_sparse:
            if not tensor.is_coalesced():
                tensor = tensor.coalesce()

            return index_sparse_with_bool(tensor, self.mask).coalesce()
        return tensor[self.mask]

    def writeat(self, target: Tensor, value):
        self.slice(target)[self.mask] = value

    @classmethod
    def from_whatever(cls, value):
        if isinstance(value, Filter):
            return value
        if isinstance(value, slice):
            return cls(slices=[value], mask=None)
        if isinstance(value, Tensor):
            return cls([], value)
        if isinstance(value, tuple | list):
            if all(isinstance(v, slice | None) for v in value):
                return cls(value, None)
            elif len(value) == 2:
                slices, mask = value
                assert isinstance(mask, Tensor | None)
                if isinstance(slices, list):
                    return cls(slices, mask)
                elif isinstance(slices, slice):
                    return cls(slices=[slices], mask=mask)
                else:
                    raise TypeError(f"Unsupported type {type(slices)}")
        else:
            raise ValueError(
                f"Unsupported type(s) for automatic conversion to Filter {value}"
            )

    def slice_attr_tensor(self, attr, default, remove_ints=False, device="cpu"):
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

    def slice_starts_tensor(self, remove_ints=False, device="cpu") -> Tensor:
        return self.slice_attr_tensor("start", 0, remove_ints, device)

    def slice_stops_tensor(self, remove_ints=False, device="cpu") -> Tensor:
        return self.slice_attr_tensor("stop", torch.inf, remove_ints, device)

    def slice_steps_tensor(self, remove_ints=False, device="cpu") -> Tensor:
        return self.slice_attr_tensor("step", 1, remove_ints, device)

    def slice_indices(self, outer_indices: Tensor, return_mask=False):
        assert all(
            [
                sl.step is None or sl.step == 1
                for sl in self.slices
                if isinstance(sl, slice)
            ]
        )
        adjustment = self.slice_starts_tensor(
            remove_ints=False, device=outer_indices.device
        ).unsqueeze(-1)
        indices = outer_indices - adjustment
        mask = torch.ones_like(outer_indices[0], dtype=torch.bool)
        for i, sl in enumerate(self.slices):
            if isinstance(sl, int):
                mask &= outer_indices[i] == sl
        mask &= (indices >= 0).all(dim=0)
        mask &= (indices % self.slice_steps_tensor().unsqueeze(-1) == 0).all(dim=0)
        rmdim = [isinstance(sl, int) for sl in self.slices] + [False] * (
            len(outer_indices.shape) - len(self.slices)
        )
        indices = torch.stack(
            [index for rm, index in zip(rmdim, indices.unbind()) if not rm]
        )
        mask &= (
            indices[: self.mask.shape[0]]
            < torch.tensor(self.mask.shape[: indices.shape[0]]).unsqueeze(-1)
        ).all(dim=0)
        if return_mask:
            return indices, mask
        if not mask.all():
            raise ValueError("Some indices out of bounds")
        return indices

    def mask_indices(self, sliced_indices):
        maskrange = torch.arange(self.mask.sum())
        n = torch.ones_like(self.mask, dtype=torch.long) * (-1)
        n[self.mask] = maskrange
        return torch.cat(
            [
                n[sliced_indices[: len(self.mask.shape)]],
                sliced_indices[len(self.mask.shape) :],
            ]
        )

    def invert_indices_slicing(self, inner_indices):
        assert all(
            [
                sl.step is None or sl.step == 1
                for sl in self.slices
                if isinstance(sl, slice)
            ]
        )
        adjustment = self.slice_starts_tensor(
            remove_ints=True, device=inner_indices.device
        ).unsqueeze(-1)
        assert (
            adjustment
            == torch.tensor(
                [
                    (0 if sl is None else sl.start)
                    for sl in self.slices
                    if not isinstance(sl, int)
                ],
                device=inner_indices.device,
                dtype=torch.long,
            ).unsqueeze(-1)
        ).all()
        indices = inner_indices + adjustment
        if any([isinstance(sl, int) for sl in self.slices]):
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

    def invert_mask(self, inner_indices):
        return torch.cat(
            [
                self.mask.nonzero()[inner_indices[0]].transpose(-2, -1),
                inner_indices[1:],
            ],
            dim=0,
        )

    def to(self, *args, **kwargs) -> "Filter":
        return Filter(
            slices=self.slices,
            mask=self._mask.to(*args, **kwargs),
            shape=self.shape,
        )

    @property
    def dimmap_i2o(self):
        return [i for i, sl in enumerate(self.slices) if isinstance(sl, slice)]

    @property
    def dimmap_o2i(self):
        return {i: o for o, i in enumerate(self.dimmap_i2o())}

    # def reduce(self, dim):
    #     slices = [sl for i, sl in enumerate(self.slices) if i != dim]
    #     shape = self.shape[:dim] + self.shape[dim + 1 :]
    #     # TODO check if mask needs reduction and assert shape=1 there
    #     # add and-reduce and or-reduce for shape != 1
    #     return Filter(slices=slices, mask=self._mask, shape=shape)


def index_sparse_with_bool(value: Tensor, mask: Tensor):
    assert value.shape[: mask.ndim] == mask.shape
    new_shape = [mask.count_nonzero()] + list(value.shape[mask.ndim :])
    include = mask[value.indices()[: mask.ndim].split(1)].squeeze()
    z = torch.zeros_like(mask, dtype=torch.long, device=value.device)
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
    value: Tensor = field(init=True, validator=validators.instance_of(Tensor))
    filter: Filter = field(
        init=True, repr=False, validator=validators.instance_of(Filter)
    )

    # @convert(filter, takes_self=True)
    # @staticmethod
    # def filter_converter(filt, inst):
    #     return Filter.from_whatever(filt)

    def __attrs_post_init__(self):
        if self.filter._mask is not None:
            assert tuple(self.value.shape) == self.filter._inner_shape()
            assert (
                self.value.shape[0] == self.filter.mask.sum()
            ), f"Value shape at dimension 0 ({self.value.shape}) does not match number of mask elements ({self.filter.mask.sum()})"
        else:
            assert True  # TODO
            # self.value.shape[0] == self.filter

    @classmethod
    def from_value_and_mask(cls, value: Tensor, mask: Tensor):
        shape = list(mask.shape) + list(value.shape[1:])
        return cls(
            value=value,
            filter=Filter(slices=[], mask=mask, shape=shape),
        )

    @classmethod
    def from_unmasked_value(cls, value: Tensor, filter: Filter, presliced=False):
        if presliced:
            return cls(value=filter.apply_mask(value), filter=filter)
        return cls(value=filter.apply(value), filter=filter)

    def mask_by_other(
        self,
        other: Union[Filter, Tensor, "FilteredTensor"],
        return_ft=False,
        presliced=None,
        value_like=False,
    ) -> Union[Tensor, "FilteredTensor"]:
        if isinstance(other, FilteredTensor):
            other = other.filter
        if isinstance(other, Filter):
            assert self.filter.shape == other.shape
            assert self.filter.slicing_tuple == other.slicing_tuple
            mask = other.mask
        if isinstance(other, Tensor):
            mask = other
            if not presliced:
                mask = self.filter.slice(mask)
        else:
            if presliced is not None:
                raise ValueError("presliced has no effect when other is not Tensor")
        if value_like:
            filtered_mask = mask
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
                shape=self.filter.shape,
            ),
        )

    def to_dense(self) -> "FilteredTensor":
        return FilteredTensor(value=self.value.to_dense(), filter=self.filter)

    def to_dense_unfiltered(self, default_value=0) -> Tensor:
        z = torch.full(self.shape, default_value, dtype=self.value.dtype)
        self.filter.slice(z)[self.filter.mask] = self.value.to_dense()
        return z

    def to_sparse_unfiltered(self, default_value=0) -> Tensor:
        # sparse z
        # values -> mask.nonzero() -> inverted slices
        ...

    def to_filtered_like_self(self, t: Tensor) -> "FilteredTensor":
        return FilteredTensor(value=t, filter=self.filter)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.filter.shape

    def cuda(self) -> "FilteredTensor":
        return self.to(torch.device("cuda"))

    def to(self, *args, **kwargs) -> "FilteredTensor":
        return FilteredTensor(
            value=self.value.to(*args, **kwargs),
            filter=self.filter.to(*args, **kwargs),
        )

    def __repr__(self):
        return f"FilteredTensor(value={self.value}, mask={self.filter})"

    def internalize_indices(self, indices: Tensor):
        assert indices.ndim == 2
        ids, mask = self.filter.slice_indices(indices, return_mask=True)
        ids = ids[:, mask]
        ids = self.filter.mask_indices(ids)
        new_mask = mask.clone()
        new_mask[mask] = ids >= 0
        ids = ids[:, (ids >= 0).all(dim=0)]
        return ids, new_mask

    def index_where_valid(self, indices: Tensor):
        ids, mask = self.internalize_indices(indices)
        return self.value[ids.unbind()], mask

    def __getitem__(self, i):
        if isinstance(i, Tensor) and i.dtype == torch.bool:
            return self.mask_by_other(i)
        if isinstance(i, Tensor) and i.dtype == torch.int64:
            si = self.filter.slice_indices(i)
            indices = self.filter.mask_indices(si)
            assert not (indices == -1).any()
            return self.value[indices.unbind()]
        # if i need this first just implement slice inversion
        for sl in self.filter.slices:
            if isinstance(sl, slice) and sl.step not in (None, 1):
                raise NotImplementedError("Step != 1 not implemented")
        # if isinstance(i, Tensor) and i.dtype == torch.int64:
        #     if i.ndim == 1:
        raise NotImplementedError()

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

    def indices(self):
        return self.filter.invert_indices_slicing(
            self.filter.invert_mask(self.value.indices())
        )

    def nonzero(self):
        return self.filter.invert_indices_slicing(
            self.filter.invert_mask(self.value.nonzero())
        )


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


def main():
    for i in range(1, 4):
        checker1([128, 128, 128, 128], i)

    value = torch.arange(20)
    slice_mask = slice(2, 14, 2)
    tensor_mask = torch.tensor([True, False, True, True, False, True])
    mask = Filter(slices=[slice_mask], mask=tensor_mask, shape=(20,))

    ft = FilteredTensor.from_unmasked_value(value, mask)

    t = torch.arange(10, 30)
    filtered_t = ft.filter.apply(t)
    result = ft.value + filtered_t

    ft_result = ft.to_filtered_like_self(result)

    dense_result = ft_result.to_dense()
    print(dense_result)


if __name__ == "__main__":
    main()
