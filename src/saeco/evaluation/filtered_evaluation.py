from pathlib import Path

import torch
from attr import define, field
from torch import Tensor
from torch.masked import MaskedTensor

# from .saved_acts import SavedActs
# from .saved_acts_config import CachingConfig

# @define
# class FilteredEvaluation(Evaluation):
#     filter: Tensor
#     filter_path: Path = field(init=False)

#     def __attrs_post_init__(self):
#         super().__attrs_post_init__()
#         self.filter_path = self.path.with_suffix(".filter")

#     def save(self):
#         super().save()
#         torch.save(self.filter, self.filter_path)

#     @classmethod
#     def create(
#         cls,


# @define
# class FilteredSaved(SavedActs):
#     filter: Tensor
#     filter_path: Path


@define
class NamedFilter:
    filter: Tensor
    filter_name: str
    # cfg: CachingConfig

    def chunk_filters(self, cfg):
        chunked = self.filter.chunk(cfg.num_chunks)
        assert len(chunked) == cfg.num_chunks
        return chunked

    def filter_chunk_output(self, output, chunk):
        return output[self.chunk_filters(chunk.cfg)[chunk.idx]]

    def save_filter(self, root_path):
        torch.save(self.filter, root_path / "filtered" / f"{self.filter_name}.pt")

    # def mask(self, tensor, chunk=None):
    #     if chunk is None:
    #         mask = self.filter
    #     else:
    #         mask = self.chunk_filters(chunk.cfg)[chunk.idx]
    #     return FilteredTensor.from_unmasked_value(tensor, mask)


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
class SliceMask:
    slices: tuple[slice, ...]
    shape: tuple[int, ...]

    def __post_init__(self):
        assert len(self.slices) == len(self.shape)
        for i, slc in enumerate(self.slices):
            assert isinstance(slc, slice)
            assert slc.start is None or slc.start >= 0 and slc.start < self.shape[i]
            assert slc.stop is None or slc.stop <= self.shape[i]
            assert slc.start is None or slc.stop is None or slc.start <= slc.stop

    def numel(self):
        n = 1
        for slc, dim in zip(self.slices, self.shape):
            step = slc.step or 1
            if slc.start is None and slc.stop is None:
                n *= dim // step
            elif slc.start is None:
                n *= slc.stop // step
            elif slc.stop is None:
                n *= (dim - slc.start) // step
            else:
                n *= (slc.stop - slc.start) // step
        return n

    def apply(self, tensor):
        if tensor.shape[: len(self.shape)] != self.shape:
            raise ValueError("Shape mismatch")
        return tensor[self.slices]


@define
class Mask:
    masks: list[Tensor | SliceMask]

    def __attrs_post_init__(self):
        # verify that the mask shapes are compatible when sequentially applied
        ...

    @property
    def shape(self): ...


@define
class FilteredTensor:
    value: Tensor
    mask: Tensor

    def __attrs_post_init__(self):
        assert self.value.shape[0] == self.mask.sum()

    @classmethod
    def from_unmasked_value(cls, value: Tensor, mask: Tensor):
        if not value.is_sparse:
            return cls(value=value[mask], mask=mask)
        new_shape = [mask.sum().item()] + list(value.shape[mask.ndim :])
        include = mask[value.indices()[: mask.ndim].split(1)].squeeze()
        z = torch.zeros_like(mask, dtype=torch.long)
        z[mask] = torch.arange(new_shape[0], device=value.device, dtype=torch.long)
        new_indices = value.indices()[:, include]
        new_indices = torch.cat(
            [z[new_indices[: mask.ndim].split(1)], new_indices[mask.ndim :]], dim=0
        )
        new_values = value.values()[include]
        return cls(
            value=torch.sparse_coo_tensor(
                indices=new_indices, values=new_values, size=new_shape
            ).coalesce(),
            mask=mask,
        )

    def mask_by_other(self, mask):
        if self.mask.ndim > mask.ndim:
            # mask = mask.unsqueeze(-1)
            mask = mask.expand(self.mask.shape)
        if self.value.is_sparse:
            if not self.value.is_coalesced():
                self.value = self.value.coalesce()
            return index_sparse_with_bool(self.value, mask[self.mask])
        return self.value[mask[self.mask]]

    @property
    def shape(self):
        return list(self.mask.shape) + list(self.value.shape[1:])

    def to_dense_masked(self):
        t = torch.empty(*self.shape, dtype=self.value.dtype, device=self.value.device)
        t[self.mask] = self.value.to_dense()
        return MaskedTensor(t, self.mask)

    def to_sparse_masked(self):
        t = torch.sparse_coo_tensor(
            indices=self.mask.nonzero().t(),
            values=self.value,
            size=self.shape,
        )
        return MaskedTensor(t, self.mask)

    def cuda(self):
        return FilteredTensor(value=self.value.cuda(), mask=self.mask.cuda())

    def to(self, *args, **kwargs):
        return FilteredTensor(
            value=self.value.to(*args, **kwargs), mask=self.mask.to(*args, **kwargs)
        )

    def to_dense(self):
        raise NotImplementedError()
        if self.mask.ndim == 1:
            return self.value.to_dense()
        return self.to_dense_masked().to_dense()
        return FilteredTensor(value=self.value.to_dense(), mask=self.mask.to_dense())

    def to_sparse_coo(self):
        return FilteredTensor(
            value=self.value.to_sparse_coo(), mask=self.mask.to_sparse_coo()
        )

    def coalesce(self):
        return FilteredTensor(value=self.value.coalesce(), mask=self.mask)

    def inverse_indices(self):
        return self.mask.nonzero().t()


def filtered_tensors_intersect(
    a: FilteredTensor, b: FilteredTensor, op
) -> FilteredTensor:
    ai = a.mask_by_other(b.mask)
    bi = b.mask_by_other(a.mask)
    return FilteredTensor(value=op(ai, bi), mask=a.mask & b.mask)


def main():
    ft1 = FilteredTensor.from_unmasked_value(
        value=torch.arange(25).reshape(5, 5).to_sparse_coo(),
        mask=torch.arange(25).reshape(5, 5) % 2 == 1,
    )
    ft2 = FilteredTensor.from_unmasked_value(
        value=torch.arange(25).reshape(5, 5),
        mask=torch.tensor([True, True, False, False, True]),
    )
    print(torch.add(ft2, ft1))
    print(torch.add(torch.arange(25).reshape(5, 5), ft1))


if __name__ == "__main__":
    main()
