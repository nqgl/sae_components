import glob
from pathlib import Path

import torch
from attrs import define, field
from pydantic import BaseModel

from .named_filter import NamedFilter

from .saved_acts_config import CachingConfig
from .storage.disk_tensor import DiskTensor
from .storage.growing_disk_tensor import GrowingDiskTensor


@define
class Artifacts:
    path: Path
    cached_config: CachingConfig
    artifacts_category: str = "artifacts"
    return_raw: bool = False

    @property
    def storage_dir(self):
        return self.path / self.artifacts_category

    def create(self, name, dtype, shape) -> DiskTensor:
        path = self.storage_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"Metadata already exists at {path}")
        return DiskTensor.create(
            path=path,
            shape=shape,
            dtype=dtype,
        )

    def __getitem__(self, name):
        if self.return_raw:
            return DiskTensor.open(self.storage_dir / name)
        return DiskTensor.open(self.storage_dir / name).tensor

    def __setitem__(self, name, value):
        assert isinstance(name, str)
        assert isinstance(value, torch.Tensor)
        assert not name in self
        disk_tensor = self.create(name, value.dtype, value.shape)
        disk_tensor.tensor[:] = value
        disk_tensor.finalize()

    def __contains__(self, name):
        return name in self.keys()

    def keys(self):
        return list(set([p.stem for p in self.storage_dir.glob("*")]))


@define
class Filters(Artifacts):
    artifacts_category: str = "filters"

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
class Metadatas(Artifacts):
    artifacts_category: str = "metadatas"

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
    seq_level: bool = False


@define
class Metadata:  # oh this maybe should just subclass DiskTensor
    storage: DiskTensor
    seq_level: bool = False
    shape: list[int] = []
    dtype: torch.dtype = torch.float32

    @property
    def tensor(self):
        return self.storage.tensor

    @classmethod
    def create(cls, path, shape, dtype, seq_level):
        return cls(
            shape=shape,
            dtype=dtype,
            seq_level=seq_level,
            storage=DiskTensor.create(path=path, shape=shape, dtype=dtype),
        )

    @classmethod
    def open(cls, path):
        dt = DiskTensor.open(path=path)
        return cls(
            shape=dt.metadata.shape,
            dtype=dt.metadata.dtype,
            seq_level=False,
            storage=dt,
        )
