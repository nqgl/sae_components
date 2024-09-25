import glob
from pathlib import Path

import torch
from attrs import define, field
from pydantic import BaseModel

from .saved_acts_config import CachingConfig
from .storage.disk_tensor import DiskTensor
from .storage.growing_disk_tensor import GrowingDiskTensor


@define
class Artifacts:
    path: Path
    cached_config: CachingConfig
    artifacts_category: str = "artifacts"

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
        return DiskTensor.open(self.storage_dir / name)

    def __setitem__(self, name, value):
        assert isinstance(value, torch.Tensor)
        assert not name in self
        disk_tensor = self.create(name, value.dtype, value.shape)
        disk_tensor.tensor[:] = value
        disk_tensor.finalize()

    def __contains__(self, name):
        path = self.storage_dir / name
        return path.exists() or path.with_suffix(".safetensors").exists()


@define
class Metadatas(Artifacts):
    artifacts_category: str = "metadatas"

    def create(self, name, dtype, seq_level=False, item_shape=[]) -> "Metadata":
        path = self.storage_dir / name
        if seq_level:
            raise NotImplementedError("seq level metadata not yet supported")
        else:
            doc_shape = [self.cached_config.num_docs]
            shape = doc_shape + item_shape
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"Metadata already exists at {path}")
        return Metadata.create(
            path,
            shape,
            dtype,
            seq_level,
        )


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
