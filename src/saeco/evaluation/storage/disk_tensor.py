from attr import define, field
from abc import ABC, abstractmethod
import torch
from pydantic import BaseModel

from saeco.misc import to_torch_dtype


class DiskTensor:
    @classmethod
    def open(cls, path):
        raise NotImplementedError


class DiskTensorMetadata(BaseModel):
    shape: list[int]
    dtype: str
    length: int
    cat_axis: int

    # cat_axis: int
    # storage_len: int
    @property
    def torch_dtype(self):
        if self.dtype == "torch.float32":
            return torch.float32
        elif self.dtype == "torch.int64":
            return torch.int64
        elif self.dtype == "torch.bool":
            return torch.bool
        elif self.dtype == "torch.float16":
            return torch.float16
        else:
            raise ValueError(f"Unsupported dtype {self.dtype}")
