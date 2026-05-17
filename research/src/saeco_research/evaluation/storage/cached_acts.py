from __future__ import annotations

from pathlib import Path

import paramsight
import torch
from attrs import define
from paramsight import get_resolved_typevars_for_base, takes_alias
from torch import Tensor

from saeco.data.dict_batch import DictBatch

from ..features import Features
from ..named_filter import NamedFilter
from .cache_config import CacheConfig
from .chunk import Chunk


@define
@paramsight.slotted_strategies.add_field
class CachedActs[InputsT: torch.Tensor | DictBatch]:
    path: Path
    cfg: CacheConfig
    chunks: list[Chunk[InputsT]]
    features: Features | None = None
    data_filter: NamedFilter | None = None

    @takes_alias
    @classmethod
    def open(cls, path: Path) -> CachedActs:
        cfg = cls._cfg_initializer(path)
        chunks = cls._chunks_initializer(path)
        return cls(
            path=path,
            cfg=cfg,
            chunks=chunks,
            features=Features.from_path(path, filter_obj=None),
        )

    @takes_alias
    @classmethod
    def _cfg_initializer(cls, path: Path) -> CacheConfig:
        return CacheConfig.model_validate_json(
            (path / CacheConfig.STANDARD_FILE_NAME).read_text()
        )

    @takes_alias
    @classmethod
    def get_inputs_type(cls) -> type[InputsT]:
        return get_resolved_typevars_for_base(cls, CachedActs)[0]  # type: ignore

    @takes_alias
    @classmethod
    def _chunks_initializer(cls, path: Path) -> list[Chunk[InputsT]]:
        return Chunk[cls.get_inputs_type()].load_chunks_from_dir(path, lazy=True)

    @takes_alias
    @classmethod
    def _filtered_chunks_initializer(
        cls, path: Path, filter_obj: NamedFilter
    ) -> list[Chunk[InputsT]]:
        return Chunk[cls.get_inputs_type()].load_chunks_from_dir(
            path=path, lazy=True, filter_obj=filter_obj
        )

    def filtered(self, filter_obj: NamedFilter) -> CachedActs[InputsT]:
        if self.data_filter is not None:
            raise ValueError("Cannot filter an already-filtered CachedActs")
        return CachedActs[self.get_inputs_type()](
            path=self.path,
            cfg=self.cfg,
            chunks=self._filtered_chunks_initializer(self.path, filter_obj),
            features=Features.from_path(self.path, filter_obj=filter_obj),
            data_filter=filter_obj,
        )

    @property
    def iter_chunks(self):
        return Chunk[self.get_inputs_type()].chunks_from_dir_iter(
            path=self.path, lazy=True
        )

    @property
    def tokens(self):
        return ChunksGetter(self, "tokens", dense_target=True, ft=True)

    @property
    def acts(self):
        return ChunksGetter(self, "acts", dense_target=False, ft=True)


def _slice_to_arange(s: slice, *, length: int) -> Tensor:
    start = 0 if s.start is None else s.start
    stop = length if s.stop is None else s.stop
    step = 1 if s.step is None else s.step
    return torch.arange(start, stop, step, dtype=torch.long)


def _select_batch(x, indices: Tensor, *, dense: bool):
    if isinstance(x, DictBatch):
        try:
            return x[indices]
        except Exception:
            return x.__class__.construct_with_other_data(
                {k: v.index_select(index=indices, dim=0) for k, v in x.items()},
                x._get_other_dict(),
            )

    if dense:
        return x.index_select(index=indices, dim=0)
    return x.index_select(index=indices, dim=0).coalesce()


@define
class ChunksGetter[InputsT: torch.Tensor | DictBatch]:
    cached_acts: CachedActs[InputsT]
    target_attr: str
    dense_target: bool
    ft: bool = False

    def get_chunk(self, i: int):
        chunk = getattr(self.cached_acts.chunks[i], self.target_attr)
        # assert isinstance(chunk, Chunk)
        return chunk

    def tokensel(self, doc_ids: Tensor, *, dense: bool) -> InputsT:
        if doc_ids.ndim != 1:
            raise ValueError("tokensel expects a 1D doc_ids tensor")

        doc_ids = doc_ids.to(dtype=torch.long, device="cpu")
        n = int(doc_ids.numel())

        sorted_ids, sort_idx = doc_ids.sort()
        inv = torch.empty_like(sort_idx)
        inv[sort_idx] = torch.arange(n, dtype=torch.long)

        cfg = self.cached_acts.cfg
        chunk_ids = sorted_ids // cfg.docs_per_chunk
        uniq, counts = chunk_ids.unique_consecutive(return_counts=True)

        parts: list[InputsT] = []
        cursor = 0
        for cid, cnt in zip(uniq.tolist(), counts.tolist(), strict=True):
            ids_chunk = sorted_ids[cursor : cursor + cnt]
            cursor += cnt

            index_for_chunk = ids_chunk if self.ft else (ids_chunk % cfg.docs_per_chunk)
            chunk_obj = self.get_chunk(cid)
            parts.append(_select_batch(chunk_obj, index_for_chunk, dense=dense))

        input_cls = self.cached_acts.get_inputs_type()

        if issubclass(input_cls, DictBatch):
            out_sorted = input_cls.cat_list(parts)
            return out_sorted.apply_func(lambda t: t.index_select(index=inv, dim=0))  # type: ignore[return-value]

        out_sorted = (
            torch.cat(parts, dim=0) if dense else torch.cat(parts, dim=0).coalesce()
        )
        if dense:
            return out_sorted.index_select(index=inv, dim=0)  # type: ignore[return-value]

        coo = out_sorted.coalesce()
        idx = coo.indices()
        new_row = sort_idx.to(device=idx.device)[idx[0]]
        new_idx = torch.cat([new_row.unsqueeze(0), idx[1:]], dim=0)
        out = torch.sparse_coo_tensor(
            new_idx, coo.values(), coo.shape, device=coo.device, dtype=coo.dtype
        ).coalesce()
        return out  # type: ignore[return-value]

    def __getitem__(self, key: slice | Tensor | int):
        cfg = self.cached_acts.cfg
        if isinstance(key, slice):
            key = _slice_to_arange(key, length=cfg.num_docs)

        if isinstance(key, int):
            key = torch.tensor([key], dtype=torch.long)

        if key.ndim == 0:
            key = key.unsqueeze(0)

        key = key.cpu()

        if key.ndim == 1:
            return self.tokensel(key, dense=self.dense_target)

        tokens = self.tokensel(key[0], dense=self.dense_target)
        if isinstance(tokens, DictBatch):
            raise TypeError(
                "Seq-level indexing not supported for DictBatch via ChunksGetter"
            )
        return tokens[(slice(None), *key[1:].tolist())]
