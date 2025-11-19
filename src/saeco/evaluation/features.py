from pathlib import Path

import torch
from attr import define
from torch import Tensor

# from .storage.filtered_chunk import FilteredChunk
from ..data.storage.sparse_growing_disk_tensor import SparseGrowingDiskTensor
from .filtered import Filter, FilteredTensor
from .named_filter import NamedFilter


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
        import time

        for i in range(10):
            try:
                feat_dir = path / "features"
                num_features = len(list(feat_dir.glob("feature*")))
                return tuple(
                    [
                        SparseGrowingDiskTensor.open(path=feat_dir / f"feature{i}")
                        for i in range(num_features)
                    ]
                )
            except FileNotFoundError:
                print(
                    "opening features failed, waiting 1 second and retrying up to 10 times"
                )
                time.sleep(1)

    def get_active(self, key):
        # for now does doc level filtering, in future with nested masks or indices could filter at token level
        if isinstance(key, Tensor | list | tuple):
            if isinstance(key, Tensor):
                assert key.dtype == torch.int64
            return [self[i] for i in key]
        if not isinstance(key, int):
            raise TypeError("need to implement handling other key type for features")
        tensor = self.feature_tensors[key].tensor.coalesce()
        slicing = [None, None, key]
        fmask = (
            self.filter.filter
            if self.filter is not None
            else torch.zeros(tensor.shape[0], dtype=torch.bool)
        )
        mask = torch.zeros_like(fmask)
        mask[tensor.indices()[0]] = 1
        mask = mask & fmask
        shape = [*tensor.shape[:2], len(self.feature_tensors)]

        return FilteredTensor.from_unmasked_value(
            value=tensor,
            filter=Filter(
                slices=slicing,
                mask=mask,
                shape=shape,
            ),
            presliced=True,
        ).to_dense()

    def __getitem__(self, key):
        if isinstance(key, Tensor | list | tuple):
            if isinstance(key, Tensor):
                assert key.dtype == torch.int64
            return [self[i] for i in key]
        if not isinstance(key, int):
            raise TypeError("need to implement handling other key type for features")
        tensor = self.feature_tensors[key].tensor
        slicing = [None, None, key]

        shape = [*tensor.shape[:2], len(self.feature_tensors)]

        return FilteredTensor.from_unmasked_value(
            value=tensor.coalesce(),
            filter=Filter(
                slices=slicing,
                mask=self.filter.filter if self.filter is not None else None,
                shape=shape,
            ),
            presliced=True,
        )
