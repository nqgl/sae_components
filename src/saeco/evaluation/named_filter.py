from __future__ import annotations

from pathlib import Path

import torch
from attrs import define, field
from torch import Tensor


@define(slots=True)
class NamedFilter:
    """
    A named boolean filter over the doc dimension.

    `filter_name=None` means it is ephemeral (cannot be persisted to a stable directory).
    """

    filter: Tensor = field()
    filter_name: str | None = field(default=None)

    @filter.validator
    def _validate_filter(self, attribute, value):
        if not isinstance(value, Tensor):
            raise TypeError("NamedFilter.filter must be a torch.Tensor")
        if value.dtype is not torch.bool:
            raise TypeError(f"NamedFilter.filter must have dtype bool, got {value.dtype}")
        if value.ndim != 1:
            raise ValueError("NamedFilter.filter must be 1D over tokens")

    def chunk_filters(self, cfg) -> list[Tensor]:
        chunked = list(self.filter.chunk(cfg.num_chunks))
        if len(chunked) != cfg.num_chunks:
            raise ValueError("Unexpected chunking behavior for filter tensor")
        return chunked

    def filter_chunk_output(self, output, chunk):
        return output[self.chunk_filters(chunk.cfg)[chunk.idx]]

    def filtered_dir(self, root_path: Path) -> Path:
        if self.filter_name is None:
            raise ValueError(
                "Current filter is unnamed. To use persistent storage on filtered evaluations, "
                "use a named, saved filter."
            )
        path = self._filtered_dir_from_root_and_name(root_path, self.filter_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def _filtered_dir_from_root_and_name(cls, root_path: Path, filter_name: str) -> Path:
        return root_path / "filtered" / filter_name