from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol, cast
from weakref import WeakValueDictionary

import torch
from attrs import define, field
from paramsight import get_resolved_typevars_for_base, takes_alias
from pydantic import BaseModel

from saeco.misc import str_to_dtype

from .compressed_safetensors import (
    CompressionType,
    load_file_compressed,
    save_file_compressed,
)


def assert_isint(x: Any) -> int:
    assert isinstance(x, int)
    return x


class DiskTensorMetadata(BaseModel):
    shape: list[int | None]
    dtype_str: str
    # cat_axis: int
    compression: CompressionType = CompressionType.NONE

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


def _numel_from_shape(shape: Sequence[int]):
    assert all(s >= 0 for s in shape)
    if len(shape) == 0:
        return 0
    size = 1
    for s in shape:
        size *= s
    return size


class RemovableOnFinalize(Protocol):
    def _remove_finalized_disktensor(
        self, key: str, disk_tensor: "DiskTensor[Any]"
    ) -> None: ...


@define
class DiskTensor[MetadataT: DiskTensorMetadata = DiskTensorMetadata]:
    path: Path
    metadata: MetadataT
    finalized: bool = False
    remove_on_finalize: WeakValueDictionary[str, RemovableOnFinalize] = field(
        factory=WeakValueDictionary
    )

    @takes_alias
    @classmethod
    def get_metadata_cls(cls) -> type[MetadataT]:
        base = get_resolved_typevars_for_base(cls, DiskTensor)[0]
        assert base is not None
        return cast("type[MetadataT]", base)

    @property
    def tensor(self) -> torch.Tensor:
        return self.view_storage(self._raw_tensor)

    @tensor.deleter
    def tensor(self):
        del self._raw_tensor

    @cached_property
    def _raw_tensor(self):
        # print("opening tensor", self.path)
        safepath = self.path.with_suffix(".safetensors")
        if safepath.exists():
            self.finalized = True
            return load_file_compressed(
                safepath, compression=self.metadata.compression
            )["finalized_tensor"]
        if self.path.exists():
            self.finalized = True
        return self._open_raw_disk_tensor(create=not self.finalized)

    @property
    def metadata_path(self) -> Path:
        return self.path.with_suffix(".metadata")

    @takes_alias
    @classmethod
    def _open_metadata(cls, path: Path) -> MetadataT:
        return cls.get_metadata_cls().model_validate_json(
            (path.with_suffix(".metadata")).read_text()
        )

    def __attrs_post_init__(self):
        if self.path.exists() or self.path.with_suffix(".safetensors").exists():
            self.finalized = True
        # self.tensor  # init tensor

    # def create_tensor(self):
    #     return self.open_disk_tensor(create=True)

    def open_disk_tensor(self, create=False):
        return self.view_storage(self._open_raw_disk_tensor(create=create))

    def _open_raw_disk_tensor(self, create=False):
        if self.path.exists() == create:
            if not self.path.exists():
                raise FileNotFoundError(
                    f"File {self.path} does not exist and create flag is not set"
                )
            else:
                raise FileExistsError(
                    f"File {self.path} already exists and create flag is set"
                )
        return torch.from_file(
            str(self.path),
            shared=True,
            size=self.numel,
            dtype=self.metadata.dtype,
        )

    def view_storage(self, storage: torch.Tensor) -> torch.Tensor:
        shape = tuple(assert_isint(s) for s in self.storage_shape)
        return storage.view(*shape)

    @classmethod
    def open(cls, path):
        return cls(
            path=path,
            metadata=cls._open_metadata(path),
        )

    @takes_alias
    @classmethod
    def create(
        cls,
        path: Path,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        compression: CompressionType = CompressionType.NONE,
    ):
        metadata = cls.get_metadata_cls()(
            shape=list(shape), dtype_str=str(dtype), compression=compression
        )

        inst = cls(path=path, metadata=metadata)
        if inst.path.exists():
            raise ValueError(f"Metadata already exists at {inst.path}")
        if inst.path.with_suffix(".safetensors").exists():
            raise ValueError(
                f"Metadata already exists at {inst.path.with_suffix('.safetensors')}"
            )
        return inst

    @property
    def storage_shape(self):
        return self.metadata.shape

    @property
    def nbytes(self):
        return self._nbytes_from_shape(self.storage_shape)

    @property
    def numel(self):
        assert not any(s is None for s in self.storage_shape)
        return _numel_from_shape(self.storage_shape)

    def _nbytes_from_shape(self, shape):
        size = self.metadata.dtype.itemsize
        for s in shape:
            size *= s
        return size

    def _save_safe(self):
        save_file_compressed(
            {"finalized_tensor": self._raw_tensor},
            self.path.with_suffix(".safetensors"),
            compression=self.metadata.compression,
        )

    def finalize(self):
        # self.resize(self.metadata.shape[self.cat_axis], truncate=True)
        self.finalized = True
        assert not self.metadata_path.exists()
        # TODO

        self._save_safe()
        if self.tensor.numel() > 0:
            self.path.unlink()

        self.metadata_path.write_text(self.metadata.model_dump_json())
        del self.tensor
        keys = list(self.remove_on_finalize.keys())
        for key in keys:
            container = self.remove_on_finalize.pop(key, None)
            if container is None:
                continue
            container._remove_finalized_disktensor(key, self)


def main():
    path = Path("testdata/storage_testing")
    dt = DiskTensor.create(path, [10, 10], torch.int64)
    dt.tensor[:] = torch.arange(100).reshape(10, 10)
    dt.finalize()
    print(dt.tensor)

    dt = DiskTensor.open(path)
    print(dt.tensor)


if __name__ == "__main__":
    main()
