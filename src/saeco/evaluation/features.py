from functools import cached_property
from pathlib import Path

import torch
from attr import define, field
from jaxtyping import Float, Int
from torch import Tensor

from .filtered import Filter, FilteredTensor, SliceMask

from .filtered_evaluation import NamedFilter
from .saved_acts_config import CachingConfig
from .storage.chunk import Chunk

# from .storage.filtered_chunk import FilteredChunk
from .storage.sparse_growing_disk_tensor import SparseGrowingDiskTensor


@define
class Features:
    # cfg: CachingConfig
    feature_tensors: tuple[SparseGrowingDiskTensor] | None = None
    filter: NamedFilter | None = None

    @classmethod
    def from_path(cls, path: Path, filter: NamedFilter | None):
        return cls(
            feature_tensors=cls._feature_tensors_initializer(path),
            filter=filter,
        )

    @classmethod
    def _feature_tensors_initializer(cls, path: Path):
        feat_dir = path / "features"
        num_features = len(list(feat_dir.glob("feature*")))
        try:
            return tuple(
                [
                    SparseGrowingDiskTensor.open(path=feat_dir / f"feature{i}")
                    for i in range(num_features)
                ]
            )
        except Exception as e:
            raise FileExistsError("Malformed features directory") from e

    def __getitem__(self, key):
        if isinstance(key, Tensor | list | tuple):
            if isinstance(key, Tensor):
                assert key.dtype == torch.int64
            return [self[i] for i in key]
        if not isinstance(key, int):
            raise TypeError("need to implement handling other key type for features")
        tensor = self.feature_tensors[key].tensor
        slicing = SliceMask(
            [slice(None), slice(None), slice(key, key + 1)],
            shape=[*tensor.shape[:2], len(self.feature_tensors)],
        )
        return FilteredTensor(
            value=tensor,
            filter=Filter(
                slices=slicing,
                mask=self.filter.filter if self.filter is not None else None,
            ),
        )
