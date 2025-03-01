import glob
from pathlib import Path

import torch
from attrs import define, field
from pydantic import BaseModel
from torch import Tensor

from saeco.evaluation.saved_acts_config import CachingConfig

from ...data.storage.disk_tensor_collection import DiskTensorCollection

from ..named_filter import NamedFilter

from ...data.storage.disk_tensor import DiskTensor
from ...data.storage.growing_disk_tensor import GrowingDiskTensor


@define(kw_only=True)
class CollectionWithCachingConfig(DiskTensorCollection):
    cached_config: CachingConfig


@define
class Artifacts(CollectionWithCachingConfig):
    stored_tensors_subdirectory_name: str = "artifacts"


@define
class Filters(CollectionWithCachingConfig):
    stored_tensors_subdirectory_name: str = "filters"

    def __setitem__(self, name, value):
        if value.shape[0] != self.cached_config.num_docs:
            raise ValueError(
                f"First dimension of filter tensor must be docs-length, got value shape {value.shape}"
            )
        if value.dtype != torch.bool:
            raise ValueError(f"Filter tensor must have dtype bool, got {value.dtype}")
        return super().__setitem__(name, value)

    def __getitem__(self, name) -> NamedFilter:
        return NamedFilter(filter=super().__getitem__(name), filter_name=name)


@define
class Metadatas(CollectionWithCachingConfig):
    stored_tensors_subdirectory_name: str = "metadatas"

    def create(self, name, dtype, item_shape=[]) -> DiskTensor:
        path = self.storage_dir / name
        doc_shape = [self.cached_config.num_docs]
        shape = doc_shape + list(item_shape)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"Metadata already exists at {path}")
        return DiskTensor.create(
            path,
            shape,
            dtype,
        )

    def get(self, name):
        return Metadata.open(self.storage_dir / name)

    def __setitem__(self, name, value):
        assert isinstance(name, str)
        assert isinstance(value, torch.Tensor)
        if name in self:
            raise ValueError(f"Metadata already exists at {name}")
        if value.shape[0] != self.cached_config.num_docs:
            raise ValueError(
                f"First dimension of metadata tensor must be docs-length, got value shape {value.shape}"
            )
        item_shape = value.shape[1:]
        disk_tensor = self.create(name=name, dtype=value.dtype, item_shape=item_shape)
        disk_tensor.tensor[:] = value
        disk_tensor.finalize()

    def set_str_translator(self, name, d):
        disk_tensor = self.get(name)
        disk_tensor.set_str_translator(d)

    def translate(self, d: dict[str, Tensor]) -> dict[str, list[str] | Tensor]:
        print(d)
        o = {
            k: m.strlist(v) if (m := self.get(k)).info.tostr is not None else v
            for k, v in d.items()
        }
        print("done")
        return o


# @define
# class Metadatas:
#     path: Path
#     cached_config: CachingConfig

#     # @classmethod
#     # def create(cls, size, dtype, doc_seq): ...

#     @property
#     def metadatas_dir(self):
#         return self.path / "metadatas"

#     def get_metadata_path(self, name):
#         return self.metadatas_dir / name

#     def create_metadata(
#         self, name, dtype, seq_level=False, item_shape=[]
#     ) -> "Metadata":
#         path = self.get_metadata_path(name)
#         if seq_level:
#             raise NotImplementedError("seq level metadata not yet supported")
#         else:
#             doc_shape = [self.cached_config.num_docs]
#             shape = doc_shape + item_shape
#         path.parent.mkdir(parents=True, exist_ok=True)
#         if path.exists():
#             raise ValueError(f"Metadata already exists at {path}")
#         return Metadata.create(
#             path,
#             shape,
#             dtype,
#             seq_level,
#         )

#     def __getitem__(self, name):
#         return Metadata.open(self.get_metadata_path(name))


class MetadataTensorInfo(BaseModel):  # TODO
    tostr: dict[int, str] | None
    fromstr: dict[str, int] | None

    @classmethod
    def open(cls, path: Path):
        path = path.with_suffix(".metadatainfo")
        if path.exists():
            return cls.model_validate_json(path.read_text())
        return cls(tostr=None, fromstr=None)

    def save(self, path: Path):
        path.with_suffix(".metadatainfo").write_text(self.model_dump_json())

    def populate(self, d):
        assert self.tostr is None and self.fromstr is None
        self.fromstr = d
        self.tostr = {v: k for k, v in d.items()}


@define
class Metadata(DiskTensor):
    info: MetadataTensorInfo | None = field(default=None)

    @classmethod
    def open(cls, path):
        return cls(
            path=path,
            metadata=cls._open_metadata(path),
            info=MetadataTensorInfo.open(path),
        )

    def finalize(self):
        self.info.save(self.path)
        return super().finalize()

    def strlist(self, tensor=None):
        tensor = tensor if tensor is not None else self.tensor
        return [self.info.tostr[i] for i in tensor.tolist()]

    def set_str_translator(self, d):
        self.info.populate(d)
        self.info.save(self.path)


# @define
# class Metadata:  # oh this maybe should just subclass DiskTensor
#     storage: DiskTensor
#     seq_level: bool = False
#     shape: list[int] = []
#     dtype: torch.dtype = torch.float32

#     @property
#     def tensor(self):
#         return self.storage.tensor

#     @classmethod
#     def create(cls, path, shape, dtype, seq_level):
#         return cls(
#             shape=shape,
#             dtype=dtype,
#             seq_level=seq_level,
#             storage=DiskTensor.create(path=path, shape=shape, dtype=dtype),
#         )

#     @classmethod
#     def open(cls, path):
#         dt = DiskTensor.open(path=path)
#         return cls(
#             shape=dt.metadata.shape,
#             dtype=dt.metadata.dtype,
#             seq_level=False,
#             storage=dt,
#         )
