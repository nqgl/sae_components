from __future__ import annotations

from pathlib import Path

import torch
from attrs import define, field
from pydantic import BaseModel
from saeco_research.evaluation.storage.cache_config import CacheConfig
from torch import Tensor

from saeco.data.storage.disk_tensor import DiskTensor
from saeco.data.storage.disk_tensor_collection import DiskTensorCollection

from ..named_filter import NamedFilter


@define(kw_only=True, slots=True)
class CollectionWithCacheConfig(DiskTensorCollection):
    cache_config: CacheConfig


@define(slots=True)
class Artifacts(CollectionWithCacheConfig):
    stored_tensors_subdirectory_name: str = "artifacts"


@define(slots=True)
class Filters(CollectionWithCacheConfig):
    stored_tensors_subdirectory_name: str = "filters"

    def __setitem__(self, name: str, value: Tensor):
        if value.shape[0] != self.cache_config.num_docs:
            raise ValueError(
                f"Filter tensor must have first dim == num_docs ({self.cache_config.num_docs}); got {value.shape}"
            )
        if value.dtype is not torch.bool:
            raise ValueError(f"Filter tensor must have dtype bool, got {value.dtype}")
        return super().__setitem__(name, value)

    def get_filter(self, name: str) -> NamedFilter:
        return NamedFilter(filter=super().__getitem__(name), filter_name=name)

    def __getitem__(self, name: str) -> NamedFilter:
        return NamedFilter(filter=super().__getitem__(name), filter_name=name)


@define(slots=True)
class Metadatas(CollectionWithCacheConfig):
    stored_tensors_subdirectory_name: str = "metadatas"

    def create(
        self,
        name: str,
        dtype: torch.dtype,
        item_shape: list[int] | tuple[int, ...] = (),
    ) -> DiskTensor:
        path = self.storage_dir / name
        shape = [self.cache_config.num_docs, *list(item_shape)]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"Metadata already exists at {path}")
        return DiskTensor.create(path, shape, dtype)

    def get(self, name: str) -> Metadata:
        return Metadata.open(self.storage_dir / name)

    def __setitem__(self, name: str, value: Tensor):
        if name in self:
            raise ValueError(f"Metadata already exists at {name}")
        if value.shape[0] != self.cache_config.num_docs:
            raise ValueError(
                f"Metadata first dim must be num_docs ({self.cache_config.num_docs}); got {value.shape}"
            )

        disk_tensor = self.create(
            name=name, dtype=value.dtype, item_shape=value.shape[1:]
        )
        disk_tensor.tensor[:] = value
        disk_tensor.finalize()

    def set_str_translator(self, name: str, d: dict[str, int]) -> None:
        self.get(name).set_str_translator(d)

    def translate(self, d: dict[str, Tensor]) -> dict[str, list[str] | Tensor]:
        out: dict[str, list[str] | Tensor] = {}
        for k, v in d.items():
            m = self.get(k)
            out[k] = m.strlist(v) if (m.info and m.info.tostr is not None) else v
        return out


class MetadataTensorInfo(BaseModel):
    tostr: dict[int, str] | None
    fromstr: dict[str, int] | None

    @classmethod
    def open(cls, path: Path) -> MetadataTensorInfo:
        info_path = path.with_suffix(".metadatainfo")
        if info_path.exists():
            return cls.model_validate_json(info_path.read_text())
        return cls(tostr=None, fromstr=None)

    def save(self, path: Path) -> None:
        path.with_suffix(".metadatainfo").write_text(self.model_dump_json())

    def populate(self, d: dict[str, int]) -> None:
        if self.tostr is not None or self.fromstr is not None:
            raise ValueError("Translator already populated")
        self.fromstr = d
        self.tostr = {v: k for k, v in d.items()}


@define(slots=True)
class Metadata(DiskTensor):
    info: MetadataTensorInfo | None = field(default=None)

    @classmethod
    def open(cls, path: Path) -> Metadata:
        return cls(
            path=path,
            metadata=cls._open_metadata(path),
            info=MetadataTensorInfo.open(path),
        )

    def finalize(self):
        if self.info is not None:
            self.info.save(self.path)
        return super().finalize()

    def strlist(self, tensor: Tensor | None = None) -> list[str]:
        if self.info is None or self.info.tostr is None:
            raise ValueError("No string translator set for this metadata")
        t = tensor if tensor is not None else self.tensor
        return [self.info.tostr[i] for i in t.tolist()]

    def set_str_translator(self, d: dict[str, int]) -> None:
        if self.info is None:
            self.info = MetadataTensorInfo(tostr=None, fromstr=None)
        self.info.populate(d)
        self.info.save(self.path)
