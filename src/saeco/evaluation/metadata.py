import glob
from pathlib import Path

import torch
from attrs import define, field
from pydantic import BaseModel

from .saved_acts_config import CachingConfig
from .storage.disk_tensor import DiskTensor
from .storage.growing_disk_tensor import GrowingDiskTensor


@define
class MetaDatas:
    path: Path
    cached_config: CachingConfig

    # @classmethod
    # def create(cls, size, dtype, doc_seq): ...

    @property
    def metadatas_dir(self):
        return self.path / "metadatas"

    def get_metadata_path(self, name):
        return self.metadatas_dir / name

    def create_metadata(self, name, dtype, seq_level=False, item_shape=[]):
        path = self.get_metadata_path(name)
        if seq_level:
            raise ValueError("seq level metadata not yet supported")
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

    def __getitem__(self, name):
        return Metadata.open(self.get_metadata_path(name))


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
            # MetadataTensorInfo.model_validate_json(
            #     (path.with_suffix(".metadata")).read_text()
            # ).seq_level,
            storage=dt,
        )
