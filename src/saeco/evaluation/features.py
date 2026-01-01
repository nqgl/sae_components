from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import overload

import torch
from attrs import define
from torch import Tensor

from ..data.storage.sparse_growing_disk_tensor import SparseGrowingDiskTensor
from .filtered import Filter, FilteredTensor
from .named_filter import NamedFilter


@define(slots=True)
class Features:
    """
    Feature activations stored as one SparseGrowingDiskTensor per feature.

    Each feature tensor typically has shape (num_docs, seq_len) and is sparse.
    We expose each feature as a FilteredTensor over a virtual tensor of shape:
        (num_docs, seq_len, num_features)
    with the last dim fixed via an int slice.
    """

    path: Path
    filter: NamedFilter | None = None

    @cached_property
    def feature_tensors(self) -> tuple[SparseGrowingDiskTensor, ...]:
        return self._open_feature_tensors(self.path)

    @classmethod
    def from_path(cls, path: Path, filter_obj: NamedFilter | None) -> "Features":
        return cls(path=path, filter=filter_obj)

    @classmethod
    def _open_feature_tensors(cls, path: Path) -> tuple[SparseGrowingDiskTensor, ...]:
        import time

        feat_dir = path / "features"
        for _ in range(10):
            try:
                if not feat_dir.exists():
                    raise FileNotFoundError(feat_dir)

                # Prefer deterministic ordering.
                candidates = sorted(
                    (p for p in feat_dir.iterdir() if p.name.startswith("feature")),
                    key=lambda p: int(p.name.removeprefix("feature")) if p.name.removeprefix("feature").isdigit() else p.name,
                )
                return tuple(SparseGrowingDiskTensor.open(path=p) for p in candidates)
            except FileNotFoundError:
                print("Opening features failed; retrying in 1s (up to 10x).")
                time.sleep(1)

        raise FileNotFoundError(f"Could not find features directory at {feat_dir}")

    def __len__(self) -> int:
        return len(self.feature_tensors)

    def get_active(self, key: int | Tensor | Sequence[int]) -> FilteredTensor | list[FilteredTensor]:
        """
        Convenience: return feature(s) with inactive tokens removed.
        """
        ft = self[key]
        if isinstance(ft, list):
            return [x.filter_inactive_docs() for x in ft]
        return ft.filter_inactive_docs()

    @overload
    def __getitem__(self, key: Tensor | Sequence[int]) -> list[FilteredTensor]: ...
    @overload
    def __getitem__(self, key: int) -> FilteredTensor: ...

    def __getitem__(self, key: int | Tensor | Sequence[int]) -> FilteredTensor | list[FilteredTensor]:
        if isinstance(key, Tensor) or not isinstance(key, int):
            if isinstance(key, Tensor):
                if key.dtype != torch.long:
                    raise TypeError("Feature id tensor must be torch.long")
                return [self[int(i.item())] for i in key]
            return [self[i] for i in key]

        tensor = self.feature_tensors[key].tensor
        if not isinstance(tensor, Tensor):
            raise TypeError("feature tensor backend did not return a torch.Tensor")

        # Virtual tensor adds a feature dimension of size len(feature_tensors).
        virtual_shape = (*tensor.shape[:2], len(self.feature_tensors))
        slicing = (None, None, int(key))

        return FilteredTensor.from_unmasked_value(
            value=tensor.coalesce() if tensor.is_sparse else tensor,
            filter_obj=Filter(
                slices=slicing,
                mask=self.filter.filter if self.filter is not None else None,
                shape=virtual_shape,
            ),
            presliced=True,
        )