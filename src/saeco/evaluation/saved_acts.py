from attr import define, field
from .saved_acts_config import CachingConfig
from pathlib import Path
from .storage.chunk import Chunk
from .storage.filtered_chunk import FilteredChunk
import torch
from functools import cached_property
from .storage.sparse_growing_disk_tensor import SparseGrowingDiskTensor
from .filtered_evaluation import Filter
from torch import Tensor
from jaxtyping import Float, Int
from .metadata import MetaDatas


@define
class SavedActs:
    path: Path
    cfg: CachingConfig
    chunks: list[Chunk]

    # filter here? or in eval?
    feature_tensors: list[SparseGrowingDiskTensor] | None = None
    data_filter: Filter | None = None

    @classmethod
    def from_path(cls, path: Path):
        cfg = cls._cfg_initializer(path)
        feature_tensors = cls._feature_tensors_initializer(path, cfg)
        chunks = cls._chunks_initializer(path)
        return cls(path=path, cfg=cfg, chunks=chunks, feature_tensors=feature_tensors)

    @classmethod
    def _cfg_initializer(cls, path: Path):
        return CachingConfig.model_validate_json(
            (path / CachingConfig.STANDARD_FILE_NAME).read_text()
        )

    @classmethod
    def _feature_tensors_initializer(cls, path: Path, cfg: CachingConfig):
        feat_dir = path / "features"
        num_features = len(list(feat_dir.glob("feature*")))
        if cfg.store_feature_tensors:
            return [
                SparseGrowingDiskTensor.open(path=feat_dir / f"feature{i}")
                for i in range(num_features)
            ]
        return None

    @classmethod
    def _chunks_initializer(cls, path: Path):
        return Chunk.load_chunks_from_dir(path, lazy=True)

    @classmethod
    def _filtered_chunks_initializer(cls, path: Path, filter):
        return FilteredChunk.load_chunks_from_dir(filter=filter, path=path, lazy=True)

    def filtered(self, filter: Filter):
        return SavedActs(
            path=self.path,
            cfg=self.cfg,
            chunks=self._filtered_chunks_initializer(self.path, filter),
            feature_tensors=None,
            data_filter=filter,
        )

    @property
    def iter_chunks(self):
        return Chunk.chunks_from_dir_iter(path=self.path, lazy=True)

    # def where_feature_active(self, feature_ids, intersection=False):
    #     l = []
    #     for c_indices, indices, values, s in self.iter_where_feature_active(
    #         feature_ids=feature_ids, intersection=intersection
    #     ):
    #         l.append(torch.sparse_coo_tensor(indices, values, s))
    #     return l

    def active_feature_tensor(self, feature_id) -> torch.Tensor:
        assert self.cfg.store_feature_tensors
        return self.feature_tensors[feature_id].tensor

    # def where_feature_active_big_tensor(self, feature_ids, intersection=False):
    #     indices_l = []
    #     values_l = []
    #     shape = None
    #     for chunk_indices, indices, values, s in self.iter_where_feature_active(
    #         feature_ids=feature_ids, intersection=intersection
    #     ):
    #         if shape is None:
    #             shape = list(s)
    #         else:
    #             # shape[0] += s[0]
    #             assert shape[1:] == list(s[1:])
    #         indices_l.append(chunk_indices)
    #         values_l.append(values)
    #     ids = torch.cat(indices_l, dim=1)
    #     values = torch.cat(values_l)
    #     return torch.sparse_coo_tensor(ids, values, [self.cfg.num_chunks] + shape)

    # def iter_where_feature_active(self, feature_ids, intersection=False):
    #     for i, chunk in enumerate(self.iter_chunks):
    #         acts = chunk.acts
    #         assert acts.is_sparse
    #         ids = acts.indices()
    #         feat_ids = ids[2]
    #         if intersection:
    #             mask = torch.ones_like(feat_ids, dtype=torch.bool)
    #         else:
    #             mask = torch.zeros_like(feat_ids, dtype=torch.bool)
    #         for feature_id in feature_ids:
    #             if intersection:
    #                 mask &= feat_ids == feature_id
    #             else:
    #                 mask |= feat_ids == feature_id
    #         if mask.any():
    #             indices = ids[:, mask]
    #             values = chunk.acts.values()[mask]
    #             chunk_indices = torch.cat(
    #                 [
    #                     torch.tensor((i), dtype=torch.int64)
    #                     .unsqueeze(0)
    #                     .expand(1, indices.shape[1]),
    #                     indices,
    #                 ],
    #                 dim=0,
    #             )
    #             yield (
    #                 chunk_indices,
    #                 indices,
    #                 values,
    #                 acts.shape,
    #             )

    @property
    def tokens(self):
        return ChunksGetter(self, "tokens", chunk_indexing=False)

    @property
    def ctokens(self):
        return ChunksGetter(self, "tokens")

    @property
    def acts(self):
        return ChunksGetter(self, "acts", chunk_indexing=False)

    # def __getitem__(self, sl: torch.Tensor):
    #     if isinstance(sl, tuple):
    #         ...
    #     if isinstance(sl, torch.Tensor):
    #         assert sl.shape[1] == 3  # indices for (doc, seq, d_dict)
    #         document = sl[:, 0]
    #         chunk_ids = document // self.cfg.docs_per_chunk
    #         document_id = document % self.cfg.docs_per_chunk
    #         tensor = torch.cat(
    #             [
    #                 self.chunks[chunk_id][
    #                     torch.cat(
    #                         [
    #                             document_id[chunk_ids == chunk_id],
    #                             sl[1:, chunk_ids == chunk_id],
    #                         ],
    #                         dim=1,
    #                     )
    #                 ]
    #                 for chunk_id in chunk_ids.unique()
    #             ]
    #         )
    #         tensor[:, 1] += chunk_ids


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


def torange(s: slice):
    kw = {}

    if s.start:
        kw["start"] = s.start
    if s.stop:
        kw["end"] = s.stop
    if s.step:
        kw["step"] = s.step
    return torch.arange(**kw)


def sparse_to_mask(t):
    v = torch.ones_like(t.values(), dtype=torch.bool)
    return torch.sparse_coo_tensor(t.indices(), v, t.shape)


# def sparse_mask_by_indices(t, indices):
def indices_to_sparse_mask(indices, shape):
    v = torch.ones(indices.shape[1], dtype=torch.bool)
    for p in range(indices.shape[0], len(shape)):
        v = v.unsqueeze(-1).expand(-1, shape[p])
    return torch.sparse_coo_tensor(indices, v, shape)


# indices_to_sparse_mask(cdoc_idx[chunk_ids == i].unsqueeze(0), (10,3))


def overlap_mask(t1, t2):
    m1 = sparse_to_mask(t1)
    m2 = sparse_to_mask(t2)
    return m1 * m2


from nqgl.mlutils.profiling.time_gpu import TimedFunc


@define
class ChunksGetter:
    saved_acts: SavedActs
    target_attr: str
    chunk_indexing: bool = True

    def get_chunk(self, i) -> Tensor:
        return getattr(self.saved_acts.chunks[i], self.target_attr)

    def document_select(self, docs: Int[Tensor, "sdoc"]):
        cdoc_idx = docs % self.saved_acts.cfg.docs_per_chunk
        chunk_ids = docs // self.saved_acts.cfg.docs_per_chunk
        chunks, i_u = chunk_ids.unique(return_inverse=True)
        return [
            self.get_chunk(chunk_id).index_select(
                dim=0, index=cdoc_idx[chunk_ids == chunk_id]
            )
            for chunk_id in chunks
        ], i_u

    def document_select(self, docs: Int[Tensor, "sdoc"]):
        cdoc_idx = docs % self.saved_acts.cfg.docs_per_chunk
        chunk_ids = docs // self.saved_acts.cfg.docs_per_chunk
        i = 2
        cdoc_idx[chunk_ids == i]
        chunks, i_u = chunk_ids.unique(return_inverse=True)
        docs_l = [
            self.get_chunk(chunk_id).index_select(
                dim=0, index=cdoc_idx[chunk_ids == chunk_id]
            )
            for chunk_id in chunks
        ]
        ndocs_l = [sel.shape[0] for sel in docs_l]
        ndocs = torch.tensor(ndocs_l, dtype=torch.int64)
        ndocs_cum = torch.cumsum(ndocs, dim=0)
        ndocs_cum = torch.cat([torch.tensor([0]), ndocs_cum])
        arange = torch.arange(ndocs.max())
        invert2doc = torch.zeros_like(docs) - 1
        for i, c in enumerate(chunks):
            mask = i_u == i
            invert2doc[mask] = arange[: mask.count_nonzero()] + ndocs_cum[i]
        assert invert2doc.numel() == invert2doc.unique().numel()
        assert invert2doc.min() == 0 and invert2doc.max() == docs.numel() - 1
        return TimedFunc(torch.cat)(docs_l)[invert2doc]

    def document_select_sparse(self, docs: Int[Tensor, "sdoc"]):

        cdoc_idx = docs % self.saved_acts.cfg.docs_per_chunk
        chunk_ids = docs // self.saved_acts.cfg.docs_per_chunk
        chunks, i_u = chunk_ids.unique(return_inverse=True)
        docs_l = [
            self.get_chunk(chunk_id)
            .index_select(dim=0, index=cdoc_idx[chunk_ids == chunk_id])
            .coalesce()
            for chunk_id in chunks
        ]
        ndocs_l = [sel.shape[0] for sel in docs_l]
        ndocs = torch.tensor(ndocs_l, dtype=torch.int64)
        ndocs_cum = torch.cumsum(ndocs, dim=0)
        ndocs_cum = torch.cat([torch.tensor([0]), ndocs_cum])
        arange = torch.arange(docs.shape[0])
        # invert2doc = torch.zeros_like(docs) - 1
        # for i, c in enumerate(chunks):
        #     mask = i_u == i
        #     invert2doc[mask] = arange[: mask.count_nonzero()] + ndocs_cum[i]
        values = []
        indices = []
        for i, (chunk_id, docs_t) in enumerate(zip(chunks, docs_l)):
            docidx = docs_t.indices()
            idx = docidx.clone()
            for j in range(docs_t.shape[0]):
                mask = docidx[0] == j
                idx[0][mask] = arange[chunk_ids == chunk_id][j]
            # cdoc_idx[chunk_ids == chunk_id]
            indices.append(idx)
            values.append(docs_t.values())
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values)
        shape = [docs.shape[0], *docs_l[0].shape[1:]]
        return torch.sparse_coo_tensor(indices, values, shape)

    def document_select_sparse_sorted(self, docs: Int[Tensor, "sdoc"]):
        cdoc_idx = docs % self.saved_acts.cfg.docs_per_chunk
        chunk_ids = docs // self.saved_acts.cfg.docs_per_chunk
        chunks = TimedFunc(chunk_ids.unique)()

        def isel(chunk_id, dim, index):
            return (
                self.get_chunk(chunk_id).index_select(dim=dim, index=index).coalesce()
            )

        isel = TimedFunc(isel)
        docs_l = [
            isel(chunk_id=chunk_id, dim=0, index=cdoc_idx[chunk_ids == chunk_id])
            for chunk_id in chunks
        ]
        return TimedFunc(torch.cat)(docs_l)
        # ndocs_l = [sel.shape[0] for sel in docs_l]
        # ndocs = torch.tensor(ndocs_l, dtype=torch.int64)
        # ndocs_cum = torch.cumsum(ndocs, dim=0)
        # ndocs_cum = torch.cat([torch.tensor([0]), ndocs_cum])
        # arange = torch.arange(docs.shape[0])
        # # invert2doc = torch.zeros_like(docs) - 1
        # # for i, c in enumerate(chunks):
        # #     mask = i_u == i
        # #     invert2doc[mask] = arange[: mask.count_nonzero()] + ndocs_cum[i]
        # values = []
        # indices = []
        # for i, (chunk_id, docs_t) in enumerate(zip(chunks, docs_l)):
        #     docidx = docs_t.indices()
        #     idx = docidx.clone()
        #     for j in range(docs_t.shape[0]):
        #         mask = docidx[0] == j
        #         idx[0][mask] = arange[mask]
        #     cdoc_idx[chunk_ids == chunk_id]
        #     indices.append(idx)
        #     values.append(docs_t.values())
        # indices = torch.cat(indices, dim=1)
        # values = torch.cat(values)
        # shape = [docs.shape[0], *docs_l[0].shape[1:]]
        # return torch.sparse_coo_tensor(indices, values, shape)

    def document_select_sparse2(self, docs: Int[Tensor, "sdoc"]):
        # just thought id check the chatgpt solution in case I was missing something easy
        # but nah this tries to allocate like 95gb haha
        cdoc_idx = docs % self.saved_acts.cfg.docs_per_chunk
        chunk_ids = docs // self.saved_acts.cfg.docs_per_chunk
        chunks = TimedFunc(chunk_ids.unique)()

        def isel(chunk_id, index):
            sparse_tensor = self.get_chunk(chunk_id)
            return sparse_tensor_select(sparse_tensor, index)

            return sparse_index_select_multidim(
                sparse_tensor, index.unsqueeze(0), dims=[0]
            )
            # mask = (sparse_tensor.indices()[0].unsqueeze(-1) == index.unsqueeze(0)).any(0)
            # values = sparse_tensor.values()[mask]
            # indices = sparse_tensor.indices()[:, mask]
            # new_indices = indices.clone()
            # new_indices[0] = torch.arange(new_indices.shape[1])
            # new_shape = [index.shape[0], *sparse_tensor.shape[1:]]
            # return torch.sparse_coo_tensor(new_indices, values, new_shape)

        isel = TimedFunc(isel)
        docs_l = [
            isel(chunk_id=chunk_id, index=cdoc_idx[chunk_ids == chunk_id])
            for chunk_id in chunks
        ]
        return TimedFunc(torch.cat)(docs_l)

    def ds(self, indices):
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
        docs = indices[0]
        cdoc_idx = docs % self.saved_acts.cfg.docs_per_chunk
        chunk_ids = docs // self.saved_acts.cfg.docs_per_chunk
        unique_chunks, chunk_inv = chunk_ids.unique(return_inverse=True)
        sel_chunks = [
            self.get_chunk(chunk_id).index_select(
                dim=0, index=cdoc_idx[chunk_ids == chunk_id]
            )
            for chunk_id in unique_chunks
        ]
        ncs = [sel.shape[0] for sel in sel_chunks]
        sel_chunks = torch.cat(sel_chunks)
        ncs = torch.tensor(ncs, dtype=torch.int64)

        new_indices = torch.cat([cdoc_idx, *indices[1:]], dim=0)

        # if indices.shape[0] >1:

    def masked_index(self, sl: torch.Tensor):
        if self.chunk_indexing:
            assert False
        if isinstance(sl, slice):
            sl = torange(sl)
        cdoc_idx = sl % self.saved_acts.cfg.docs_per_chunk
        chunk_ids = sl // self.saved_acts.cfg.docs_per_chunk
        if isinstance(sl, int):
            return getattr(self.saved_acts.chunks[chunk_ids], self.target_attr)[
                cdoc_idx
            ]
        else:
            assert isinstance(sl, torch.Tensor)
            for chunk_id in range(len(self.saved_acts.chunks)):
                cmask = chunk_ids == chunk_id
                if not cmask.any():
                    continue
                chunk_tensor = getattr(
                    self.saved_acts.chunks[chunk_id], self.target_attr
                )
                ids = sl[cmask]
                chunk_indices = torch.cat(
                    [
                        cdoc_idx[cmask].unsqueeze(0),
                        ids[1:],
                    ],
                    dim=0,
                )
                overlap = overlap_mask(chunk_tensor, chunk_indices)
                # values =

        #         ns = shape[ids.ndim :]
        #         values = chunk[0]
        # return torch.cat(
        #     [
        #         getattr(
        #             self.saved_acts.chunks[chunk_id], self.target_attr
        #         ).index_select(dim=0, index=cdoc_idx[chunk_ids == chunk_id])
        #         for chunk_id in chunks
        #     ]
        # )

    def masked_index_of_sparse(self, sl: torch.Tensor):
        if self.chunk_indexing:
            assert False
        if isinstance(sl, slice):
            sl = torange(sl)
        cdoc_idx = sl[0] % self.saved_acts.cfg.docs_per_chunk
        chunk_ids = sl[0] // self.saved_acts.cfg.docs_per_chunk
        if isinstance(sl, int):
            assert False
        else:
            mask_ids = []
            content_ids = []
            content_values = []
            for chunk_id in range(len(self.saved_acts.chunks)):
                chunk_mask = chunk_ids == chunk_id
                if not chunk_mask.any():
                    continue
                chunk: Chunk = getattr(
                    self.saved_acts.chunks[chunk_id], self.target_attr
                )
                ids = sl[:, chunk_mask]
                doc_ids = torch.cat(
                    [
                        cdoc_idx[chunk_mask].unsqueeze(0),
                        ids[1:],
                    ],
                    dim=0,
                )
                t = chunk[doc_ids]
                indices = t.indices()
                indices[0] += chunk_id * self.saved_acts.cfg.docs_per_chunk
                mask_ids.append(indices)
                content_ids.append(indices)
                content_values.append(t.values())
            mask_ids = torch.cat(mask_ids, dim=1)
            content_ids = torch.cat(content_ids, dim=1)
            content_values = torch.cat(content_values)
            shape = [
                self.saved_acts.cfg.num_chunks * self.saved_acts.cfg.docs_per_chunk,
                *t.shape[1:],
            ]
            return torch.sparse_coo_tensor(mask_ids, content_values, shape)

    def document_index(self, sl: torch.Tensor):
        if self.chunk_indexing:
            assert False
        if isinstance(sl, slice):
            sl = torange(sl)
        cdoc_idx = sl % self.saved_acts.cfg.docs_per_chunk
        chunk_ids = sl // self.saved_acts.cfg.docs_per_chunk
        if isinstance(sl, int):
            return getattr(self.saved_acts.chunks[chunk_ids], self.target_attr)[
                cdoc_idx
            ]
        else:
            assert isinstance(sl, torch.Tensor)
            chunks = chunk_ids.unique()
        return torch.cat(
            [
                getattr(
                    self.saved_acts.chunks[chunk_id], self.target_attr
                ).index_select(dim=0, index=cdoc_idx[chunk_ids == chunk_id])
                for chunk_id in chunks
            ]
        )

    def __getitem__(self, sl: torch.Tensor):
        if not isinstance(sl, tuple):
            return self.document_index(sl)
        return self.document_index(sl[0])[
            :, sl[1:]
        ]  # slower than needs to be but mostly slow for tokens and those are prob not bottleneck ever

        # not set in stone what approach to use so keeping old code around for now:
        if not self.chunk_indexing:
            if isinstance(sl, tuple | list):
                if sl[0] == slice(None, None, None):
                    chunks = range(len(self.saved_acts.chunks))
                    ...
                else:
                    chunk_ids = sl[0] // self.saved_acts.cfg.docs_per_chunk
                    cdoc_idx = sl[0] % self.saved_acts.cfg.docs_per_chunk
                    chunks = chunk_ids.unique()
                from_chunks = []
                for chunk_id in chunks:

                    chunk = getattr(self.saved_acts.chunks[chunk_id], self.target_attr)

                    [
                        torch.stack([cdoc_idx, *sl[1:]])[
                            :, chunk_ids == chunk_id
                        ].tolist()
                    ]

                return torch.cat(from_chunks)
        if isinstance(sl, slice):
            sl = (sl,)

        if isinstance(sl, tuple | list):
            sl = [(torange(s) if isinstance(s, slice) else s) for s in sl]
            sl = [
                torch.tensor([i], dtype=torch.int64) if isinstance(i, int) else i
                for i in sl
            ]
            if all([isinstance(i, torch.Tensor) for i in sl]):
                sl = torch.stack(sl, dim=0)
        if isinstance(sl, torch.Tensor):
            # assert sl.dtype == torch.int64
            # this seems like it would be super slow, I'm not a fan
            if self.chunk_indexing:
                chunk_ids = sl[0:1]
                return [
                    getattr(self.saved_acts.chunks[chunk_id], self.target_attr)[
                        sl[1:][chunk_ids == chunk_id]
                    ]
                    for chunk_id in chunk_ids.unique()
                ]

            else:
                chunk_ids = sl[0] // self.saved_acts.cfg.docs_per_chunk
                cdoc_idx = sl[0] % self.saved_acts.cfg.docs_per_chunk

                return torch.cat(
                    [
                        getattr(self.saved_acts.chunks[chunk_id], self.target_attr)[
                            torch.stack([cdoc_idx, *sl[1:]])[
                                :, chunk_ids == chunk_id
                            ].tolist()
                        ]
                        for chunk_id in chunk_ids.unique()
                    ]
                )

                return torch.cat(
                    [
                        getattr(self.saved_acts.chunks[chunk_id], self.target_attr)[
                            torch.cat([cdoc_idx, *sl[1:]])[chunk_ids == chunk_id]
                        ]
                        for chunk_id in chunk_ids.unique()
                    ]
                )

            if False:
                chunk_ids = sl[0] // self.saved_acts.cfg.docs_per_chunk
                cdoc_idx = sl[0] % self.saved_acts.cfg.docs_per_chunk
                return torch.cat(
                    [
                        getattr(self.saved_acts.chunks[chunk_id], self.target_attr)[
                            [
                                i  # unholy
                                for i in torch.cat([cdoc_idx.unsqueeze(0), sl[1:]])[
                                    :, chunk_ids == chunk_id
                                ]
                            ]
                        ]
                        for chunk_id in chunk_ids.unique()
                    ]
                )

    @property
    def ndoc(self):
        return len(self.saved_acts.chunks) * self.saved_acts.cfg.docs_per_chunk


import torch
from functorch import vmap


# def sparse_tensor_select(sparse_tensor, index):
#     # Map multidimensional indices to linear indices for efficient comparison
#     def map_to_linear_indices(indices, shape):
#         # Compute strides for each dimension
#         strides = torch.tensor(shape[1:]).flip(0).cumprod(0).flip(0)
#         strides = torch.cat((strides, torch.tensor([1]))).to(indices.device)
#         # Calculate linear indices
#         linear_indices = (indices * strides.unsqueeze(1)).sum(0)
#         return linear_indices

#     # Get the shape of the sparse tensor
#     tensor_shape = sparse_tensor.shape

#     # Map the sparse tensor's indices and the target indices to linear indices
#     sparse_linear_indices = map_to_linear_indices(sparse_tensor.indices(), tensor_shape)
#     index_linear_indices = map_to_linear_indices(index.T, tensor_shape)

#     # Use torch.isin for efficient membership testing
#     mask = torch.isin(sparse_linear_indices, index_linear_indices)

#     # Select the values and indices based on the mask
#     values = sparse_tensor.values()[mask]
#     indices = sparse_tensor.indices()[:, mask]

#     # Adjust the indices for the new tensor
#     new_indices = indices.clone()
#     new_indices[0] = torch.arange(new_indices.shape[1])

#     # Define the new shape
#     new_shape = [index.shape[0], *tensor_shape[1:]]

#     # Create the new sparse tensor
#     return torch.sparse_coo_tensor(new_indices, values, new_shape)


def sparse_tensor_select(sparse_tensor, index, keep_shape=True):
    # Map multidimensional indices to linear indices for efficient comparison

    # Get the shape of the sparse tensor
    tensor_shape = sparse_tensor.shape

    # Use torch.isin for efficient membership testing
    mask = torch.isin(sparse_tensor.indices()[0], index)

    # Select the values and indices based on the mask
    values = sparse_tensor.values()[mask]
    indices = sparse_tensor.indices()[:, mask]

    # Adjust the indices for the new tensor

    # Create the new sparse tensor
    if keep_shape:
        new_indices = indices.clone()
        return torch.sparse_coo_tensor(new_indices, values, tensor_shape)

    new_shape = [index.shape[0], *tensor_shape[1:]]
    imap = dict(zip(index.unsqueeze(-1), range(index.shape[0])))
    import functorch

    new_indices = torch.vmap(lambda x: imap[x], chunk_size=1)(indices[0])
    return torch.sparse_coo_tensor(new_indices, values, new_shape)


import torch

import torch


def sparse_index_select_multidim(sparse_tensor, index_tensor, dims):
    """
    Selects elements from a sparse tensor based on multidimensional indices over specified dimensions.

    Parameters:
    - sparse_tensor: torch.sparse_coo_tensor
    - index_tensor: torch.LongTensor of shape (len(dims), n_indices)
    - dims: List or tuple of dimensions over which we are indexing

    Returns:
    - new_sparse_tensor: torch.sparse_coo_tensor
    """
    device = sparse_tensor.device
    dtype = sparse_tensor.dtype
    sparse_indices = sparse_tensor.indices()
    sparse_values = sparse_tensor.values()
    nnz = sparse_values.shape[0]
    tensor_shape = sparse_tensor.shape
    dims = list(dims)  # Ensure dims is a list

    # Compute strides for the specified dimensions to create unique keys
    strides = torch.ones(len(dims), dtype=torch.long, device=device)
    for i in range(len(dims) - 2, -1, -1):
        strides[i] = strides[i + 1] * tensor_shape[dims[i + 1]]

    # Extract the relevant indices from sparse_indices
    sparse_indices_selected = sparse_indices.index_select(
        0, torch.tensor(dims, device=device)
    )
    # Compute keys for sparse_indices and index_tensor
    sparse_keys = (sparse_indices_selected * strides.unsqueeze(1)).sum(dim=0)
    index_keys = (index_tensor * strides.unsqueeze(1)).sum(dim=0)

    # Ensure index_keys are unique and sorted
    index_keys, index_perm = index_keys.sort()
    index_tensor_sorted = index_tensor[:, index_perm]

    # Find matching indices using torch.isin
    mask = torch.isin(sparse_keys, index_keys)

    # Select the matching values and indices
    values = sparse_values[mask]
    indices = sparse_indices[:, mask]  # Keep all dimensions
    matched_sparse_keys = sparse_keys[mask]

    # Map matched_sparse_keys to new indices (positions in index_keys)
    positions = torch.searchsorted(index_keys, matched_sparse_keys)
    # Confirm that positions are valid
    valid = index_keys[positions] == matched_sparse_keys
    if not valid.all():
        raise ValueError("Mismatch in positions found during searchsorted.")

    # Adjust the indices for the dimensions we are indexing over
    new_indices = indices.clone()
    # For the dimensions we are indexing over, map the old indices to new indices
    for idx, dim in enumerate(dims):
        old_indices = indices[dim]
        # Create a mapping from the original indices to their new positions
        unique_indices = index_tensor_sorted[idx].unique(sorted=True)
        index_mapping = {
            orig_idx.item(): new_idx for new_idx, orig_idx in enumerate(unique_indices)
        }
        new_indices_dim = torch.tensor(
            [index_mapping.get(idx.item(), -1) for idx in old_indices],
            dtype=torch.long,
            device=device,
        )
        # Filter out indices that were not in index_mapping (if any)
        valid_mask = new_indices_dim != -1
        new_indices = new_indices[:, valid_mask]
        values = values[valid_mask]
        new_indices[dim] = new_indices_dim[valid_mask]

    # Define the new shape
    new_shape = list(sparse_tensor.shape)
    for idx, dim in enumerate(dims):
        new_shape[dim] = index_tensor_sorted[idx].unique().numel()

    return torch.sparse_coo_tensor(
        new_indices, values, new_shape, dtype=dtype, device=device
    )
