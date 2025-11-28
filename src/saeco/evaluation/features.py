from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import overload

import torch
from attrs import define
from torch import Tensor

# from .storage.filtered_chunk import FilteredChunk
from ..data.storage.sparse_growing_disk_tensor import SparseGrowingDiskTensor
from .filtered import Filter, FilteredTensor
from .named_filter import NamedFilter


@define
class Features:
    path: Path
    # cfg: CachingConfig
    # feature_tensors: tuple[SparseGrowingDiskTensor] | None = None
    filter: NamedFilter | None = None

    @cached_property
    def feature_tensors(self) -> tuple[SparseGrowingDiskTensor, ...]:
        return self._feature_tensors_initializer(self.path)

    @classmethod
    def from_path(cls, path: Path, filter_obj: NamedFilter | None):
        return cls(
            path=path,
            filter=filter_obj,
        )

    @classmethod
    def _feature_tensors_initializer(
        cls, path: Path
    ) -> tuple[SparseGrowingDiskTensor, ...]:
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
                    "opening features failed, waiting 1 second",
                    "and retrying up to 10 times",
                )
                time.sleep(1)
        raise FileNotFoundError(f"Could not find features at {path}")

    def get_active(  # TODO oh, it's not even used.
        self, key: int | Tensor | Sequence[int]
    ) -> FilteredTensor | list[FilteredTensor]:
        # for now does doc level filtering, in future with nested masks
        # or indices could filter at token level
        if isinstance(key, Tensor) or not isinstance(key, int):
            return self[key]  # TODO is this code path okay
            # seems like it may be doing a totally different thing
            ### previously:
            # if isinstance(key, Tensor):
            #     assert key.dtype == torch.int64
            #     [self[i] for i in key]
            # return [self[i] for i in key]
            # I wonder if the above should be swappred to calls to self.get_active(i)?
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
            filter_obj=Filter(
                slices=slicing,
                mask=mask,
                virtual_shape=shape,
            ),
            presliced=True,
        ).to_dense()

    @overload
    def __getitem__(self, key: Tensor | Sequence[int]) -> list[FilteredTensor]: ...
    @overload
    def __getitem__(self, key: int) -> FilteredTensor: ...

    def __getitem__(
        self, key: int | Tensor | Sequence[int]
    ) -> FilteredTensor | list[FilteredTensor]:
        if isinstance(key, Tensor) or not isinstance(key, int):
            if isinstance(key, Tensor):
                assert key.dtype == torch.int64
                return [self[int(i.item())] for i in key]
            return [self[i] for i in key]
        if not isinstance(key, int):
            raise TypeError("need to implement handling other key type for features")
        tensor = self.feature_tensors[key].tensor
        slicing = [None, None, key]

        shape = [*tensor.shape[:2], len(self.feature_tensors)]

        return FilteredTensor.from_unmasked_value(
            value=tensor.coalesce(),
            filter_obj=Filter(
                slices=slicing,
                mask=self.filter.filter if self.filter is not None else None,
                virtual_shape=tuple(shape),
            ),
            presliced=True,
        )
