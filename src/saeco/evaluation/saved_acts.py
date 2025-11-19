from pathlib import Path

import torch
from attr import define
from jaxtyping import Int
from torch import Tensor

from .features import Features
from .named_filter import NamedFilter
from .saved_acts_config import CachingConfig
from .storage.chunk import Chunk


@define
class SavedActs:
    path: Path
    cfg: CachingConfig
    chunks: list[Chunk]
    features: Features | None = None
    data_filter: NamedFilter | None = None

    @classmethod
    def from_path(cls, path: Path):
        cfg = cls._cfg_initializer(path)
        chunks = cls._chunks_initializer(path)
        return cls(
            path=path,
            cfg=cfg,
            chunks=chunks,
            features=Features.from_path(path, filter=None),
        )

    @classmethod
    def _cfg_initializer(cls, path: Path):
        return CachingConfig.model_validate_json(
            (path / CachingConfig.STANDARD_FILE_NAME).read_text()
        )

    @classmethod
    def _chunks_initializer(cls, path: Path):
        return Chunk.load_chunks_from_dir(path, lazy=True)

    @classmethod
    def _filtered_chunks_initializer(cls, path: Path, filter):
        return Chunk.load_chunks_from_dir(filter=filter, path=path, lazy=True)

    def filtered(self, filter: NamedFilter):
        if self.data_filter is not None:
            raise ValueError(
                "Tried to add a filter to already filtered dataset. Add filters to root (unfiltered) dataset instead"
            )
        return SavedActs(
            path=self.path,
            cfg=self.cfg,
            chunks=self._filtered_chunks_initializer(self.path, filter),
            features=Features.from_path(self.path, filter=filter),
            data_filter=filter,
        )

    @property
    def iter_chunks(self):
        return Chunk.chunks_from_dir_iter(path=self.path, lazy=True)

    @property
    def tokens(self):
        return ChunksGetter(self, "tokens", dense_target=True, ft=True)
        return ChunksGetter(self, "tokens_raw", dense_target=True)

    @property
    def acts(self):
        return ChunksGetter(self, "acts", dense_target=False, ft=True)
        return ChunksGetter(self, "acts_raw", dense_target=False)

    @property
    def filtered_acts(self):
        return ChunksGetter(self, "acts")

    @property
    def filtered_tokens(self):
        return ChunksGetter(self, "tokens")


def torange(s: slice):
    kw = {}

    if s.start:
        kw["start"] = s.start or 0
    if s.stop:
        kw["end"] = s.stop
    if s.step:
        kw["step"] = s.step
    return torch.arange(**kw)


@define
class ChunksGetter:
    saved_acts: SavedActs
    target_attr: str
    dense_target: bool
    ft: bool = False

    def get_chunk(self, i) -> Tensor:
        return getattr(self.saved_acts.chunks[i], self.target_attr)

    def docsel(self, doc_ids: Int[Tensor, "sdoc"], dense=True):
        assert doc_ids.ndim == 1
        sdi = doc_ids.argsort(descending=False)
        doc_ids = doc_ids[sdi]
        if self.ft:
            cdoc_idx = doc_ids
        else:
            cdoc_idx = doc_ids % self.saved_acts.cfg.docs_per_chunk
        chunk_ids = doc_ids // self.saved_acts.cfg.docs_per_chunk
        chunks = chunk_ids.unique()

        def isel(chunk_id, dim, index):
            if dense:
                return self.get_chunk(chunk_id).index_select(dim=dim, index=index)
            return (
                self.get_chunk(chunk_id).index_select(dim=dim, index=index).coalesce()
            )

        docs_l = [
            isel(chunk_id=chunk_id, dim=0, index=cdoc_idx[chunk_ids == chunk_id])
            for chunk_id in chunks
        ]
        sorted_out = torch.cat(docs_l)
        if dense:
            out = torch.empty_like(sorted_out)
            out[sdi] = sorted_out
        else:
            sorted_out = sorted_out.coalesce()
            oi = sorted_out.indices().clone()
            oi[0][sdi[oi[0]]] = sorted_out.indices()[0]
            out = torch.sparse_coo_tensor(
                oi,
                sorted_out.values(),
                sorted_out.shape,
            ).coalesce()
        return out

    def __getitem__(self, sl: torch.Tensor) -> Tensor:
        if isinstance(sl, slice):
            sl = torange(sl)
        if sl.ndim == 0:
            sl = sl.unsqueeze(0)
        sl = sl.cpu()
        if sl.ndim == 1:
            return self.docsel(sl, dense=self.dense_target)
        print(
            "warning: performing seq level indexing, which is inefficient for accesses which index multiple tokens per document"
        )
        return self.docsel(sl[0], dense=self.dense_target)[:, sl[1:]]
