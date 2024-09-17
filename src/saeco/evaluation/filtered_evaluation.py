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
class Filter:
    filter: Tensor
    filter_name: str
    # cfg: CachingConfig

    def chunk_filters(self, cfg):
        chunked = self.filter.chunk(cfg.num_chunks)
        assert len(chunked) == cfg.num_chunks
        return chunked

    def mask(self, tensor, chunk=None):
        if chunk is None:
            mask = self.filter
        else:
            mask = self.chunk_filters(chunk.cfg)[chunk.idx]
        # if tensor.is_sparse:
        #     mask = mask.to_sparse_coo()
        #     if mask.shape != tensor.shape:
        #         xv = mask.values()
        #         for i in range(tensor.ndim - mask.ndim):
        #             xv = xv.unsqueeze(-1)
        #         expand = [-1] * mask.ndim
        #         mask = torch.sparse_coo_tensor(
        #             indices=mask.indices(),
        #             values=xv.expand(expand + list(tensor.shape[mask.ndim :])),
        #             size=tensor.shape,
        #         )
        return FilteredTensor.from_unmasked_value(tensor, mask)


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
    mask: Tensor

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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        assert False
        if kwargs is None:
            kwargs = {}
        t_args = []
        f_args = []
        allargs = list(args) + list(kwargs.values())

        def rebuild_ak():
            nargs = allargs[: len(args)]
            nkwargs = {k: allargs[len(args) + i] for i, k in enumerate(kwargs.keys())}
            return nargs, nkwargs

        for i, arg in enumerate(allargs):
            if isinstance(arg, FilteredTensor):
                f_args.append(i)
            elif isinstance(arg, Tensor):
                t_args.append(i)
        if len(f_args) + len(t_args) != 2:
            return NotImplemented
        if len(t_args) == 1:
            t = args[t_args[0]]
            f = args[f_args[0]]
            as_filtered_values = t[f.mask]
            allargs[t_args[0]] = as_filtered_values
            allargs[f_args[0]] = f.value
            nargs, nkwargs = rebuild_ak()
            res = func(*nargs, **nkwargs)
            return FilteredTensor(value=res, mask=f.mask)
        elif len(f_args) == 2:
            a = args[f_args[0]]
            b = args[f_args[1]]
            if len(args) == 2 and len(kwargs) == 0:
                return intersect(a, b, func)
            ai = a.mask_by_other(b.mask)

    def cuda(self):
        return FilteredTensor(value=self.value.cuda(), mask=self.mask.cuda())

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


def intersect(a: FilteredTensor, b: FilteredTensor, op) -> FilteredTensor:
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
