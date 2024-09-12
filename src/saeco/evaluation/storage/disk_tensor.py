from attr import define, field
from abc import ABC, abstractmethod
import torch
from pydantic import BaseModel
from pathlib import Path
from saeco.misc import str_to_dtype


class DiskTensorMetadata(BaseModel):
    shape: list[int]
    dtype_str: str
    # cat_axis: int

    @property
    def dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype_str)

    @dtype.setter
    def dtype(self, value):
        self.dtype_str = str(value)


def tensorclass(dtype):
    if dtype == torch.float32:
        return torch.FloatTensor
    elif dtype == torch.int64:
        return torch.LongTensor
    elif dtype == torch.bool:
        return torch.BoolTensor
    elif dtype == torch.float16:
        return torch.HalfTensor
    elif dtype == torch.bfloat16:
        return torch.BFloat16Tensor
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


@define
class DiskTensor:
    path: Path
    metadata: DiskTensorMetadata
    tensor: torch.Tensor = field(init=False)
    finalized: bool = False

    @tensor.default
    def _tensor_default(self):
        if self.path.exists():
            self.finalized = True
        return self.open_disk_tensor()

    def create_tensor(self):
        return self.open_disk_tensor(create=True)

    def open_disk_tensor(self, create=False):
        if self.path.exists() == create:
            if not self.path.exists():
                raise FileNotFoundError(
                    f"File {self.path} does not exist and create flag is not set"
                )
            else:
                raise FileExistsError(
                    f"File {self.path} already exists and create flag is set"
                )
        return tensorclass(self.metadata.dtype)(
            torch.UntypedStorage.from_file(
                str(self.path),
                shared=True,
                nbytes=self.nbytes,
            )
        ).reshape(
            *[
                s if i != self.cat_axis else self.storage_len
                for i, s in enumerate(self.metadata.shape)
            ]
        )

    @classmethod
    def open(cls, path):
        raise NotImplementedError

    @classmethod
    def create(cls, path, shape, dtype, initial_nnz=4096, cat_axis=0):
        raise NotImplementedError

    @property
    def storage_shape(self):
        return self.metadata.shape

    @property
    def nbytes(self):
        return self._nbytes_from_shape(self.storage_shape)

    def _nbytes_from_shape(self, shape):
        size = self.metadata.dtype.itemsize
        for s in shape:
            size *= s
        return size

    @property
    def metadata_path(self):
        return self.path.with_suffix(".metadata")

    def finalize(self):
        # self.resize(self.metadata.shape[self.cat_axis], truncate=True)
        self.finalized = True
        assert not self.metadata_path.exists()
        self.metadata_path.write_text(self.metadata.model_dump_json())
