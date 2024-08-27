from attr import define, field
from saeco.evaluation.saved_acts_config import CachingConfig
from pathlib import Path
from saeco.evaluation.chunk import Chunk
import torch
from functools import cached_property
from saeco.evaluation.sparse_growing_disk_tensor import SparseGrowingDiskTensor


@define
class SavedActs:
    path: Path
    chunks: list[Chunk] = field(init=False)
    num_chunks: int = field(init=False)
    feature_tensors: list[SparseGrowingDiskTensor] | None = field(init=False)

    # cfg: CachingConfig
    @cached_property
    def cfg(self):
        return CachingConfig.model_validate_json(
            (self.path / CachingConfig.STANDARD_FILE_NAME).read_text()
        )

    @feature_tensors.default
    def _feature_tensors_initializer(self):
        feat_dir = self.path / "features"
        num_features = len(list(feat_dir.glob("feature*")))
        if self.cfg.store_feature_tensors:
            return [
                SparseGrowingDiskTensor.open(path=feat_dir / f"feature{i}")
                for i in range(num_features)
            ]
        return None

    @chunks.default
    def _chunks_initializer(self):
        return Chunk.load_chunks_from_dir(self.path, lazy=True)

    @num_chunks.default
    def _num_chunks_initializer(self):
        return len(Chunk.load_chunks_from_dir(self.path, lazy=True))

    @property
    def iter_chunks(self):
        return Chunk.chunks_from_dir_iter(path=self.path, lazy=True)

    def where_feature_active(self, feature_ids, intersection=False):
        l = []
        for c_indices, indices, values, s in self.iter_where_feature_active(
            feature_ids=feature_ids, intersection=intersection
        ):
            l.append(torch.sparse_coo_tensor(indices, values, s))
        return l

    def active_feature_tensor(self, feature_id) -> torch.Tensor:
        assert self.cfg.store_feature_tensors
        return self.feature_tensors[feature_id].tensor

    def where_feature_active_big_tensor(self, feature_ids, intersection=False):
        indices_l = []
        values_l = []
        shape = None
        for chunk_indices, indices, values, s in self.iter_where_feature_active(
            feature_ids=feature_ids, intersection=intersection
        ):
            if shape is None:
                shape = list(s)
            else:
                # shape[0] += s[0]
                assert shape[1:] == list(s[1:])
            indices_l.append(chunk_indices)
            values_l.append(values)
        ids = torch.cat(indices_l, dim=1)
        values = torch.cat(values_l)
        return torch.sparse_coo_tensor(ids, values, [self.cfg.num_chunks] + shape)

    def iter_where_feature_active(self, feature_ids, intersection=False):
        for i, chunk in enumerate(self.iter_chunks):
            acts = chunk.acts
            assert acts.is_sparse
            ids = acts.indices()
            feat_ids = ids[2]
            if intersection:
                mask = torch.ones_like(feat_ids, dtype=torch.bool)
            else:
                mask = torch.zeros_like(feat_ids, dtype=torch.bool)
            for feature_id in feature_ids:
                if intersection:
                    mask &= feat_ids == feature_id
                else:
                    mask |= feat_ids == feature_id
            if mask.any():
                indices = ids[:, mask]
                values = chunk.acts.values()[mask]
                chunk_indices = torch.cat(
                    [
                        torch.tensor((i), dtype=torch.int64)
                        .unsqueeze(0)
                        .expand(1, indices.shape[1]),
                        indices,
                    ],
                    dim=0,
                )
                yield (
                    chunk_indices,
                    indices,
                    values,
                    acts.shape,
                )

    def __getitem__(self, sl):
        if isinstance(sl, tuple):
            ...
        if isinstance(sl, torch.Tensor):
            assert sl.shape[1] == 3  # indices for (doc, seq, d_dict)
            document = sl[:, 0]
            chunk_ids = document // self.cfg.docs_per_chunk
            document_id = document % self.cfg.docs_per_chunk
            tensor = torch.cat(
                [
                    self.chunks[chunk_id][
                        torch.cat(
                            [
                                document_id[chunk_ids == chunk_id],
                                sl[1:, chunk_ids == chunk_id],
                            ],
                            dim=1,
                        )
                    ]
                    for chunk_id in chunk_ids.unique()
                ]
            )
            tensor[:, 1] += chunk_ids

    @property
    def tokens(self):
        return ChunksGetter(self, "tokens", chunk_indexing=False)

    @property
    def ctokens(self):
        return ChunksGetter(self, "tokens")

    @property
    def acts(self):
        return ChunksGetter(self, "acts")


@define
class ChunksGetter:
    saved_acts: SavedActs
    target_attr: str
    chunk_indexing: bool = True

    def __getitem__(self, sl):
        if isinstance(sl, tuple):
            sl = torch.stack((sl), dim=0)
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
                            torch.cat([cdoc_idx, *sl[1:]])[chunk_ids == chunk_id]
                        ]
                        for chunk_id in chunk_ids.unique()
                    ]
                )
