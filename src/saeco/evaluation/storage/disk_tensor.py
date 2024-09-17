from attr import define, field
from abc import ABC, abstractmethod
import torch
from pydantic import BaseModel
from pathlib import Path
from saeco.misc import str_to_dtype
from safetensors.torch import save_file, load_file


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
    finalized: bool = False
    tensor: torch.Tensor = field(init=False, default=None)

    # @tensor.default
    def _tensor_default(self):
        safepath = self.path.with_suffix(".safetensors")
        if safepath.exists():
            self.finalized = True
            return load_file(safepath)["finalized_tensor"]
        if self.path.exists():
            self.finalized = True
        return self.open_disk_tensor(create=not self.finalized)

    @classmethod
    def _open_metdata(cls, path: Path):
        return DiskTensorMetadata.model_validate_json(
            (path.with_suffix(".metadata")).read_text()
        )

    def __attrs_post_init__(self):
        assert self.tensor is None
        self.tensor = self._tensor_default()

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
        ).reshape(self.storage_shape)

    @classmethod
    def open(cls, path):
        return cls(
            path=path,
            metadata=cls._open_metdata(path),
        )

    @classmethod
    def create(cls, path, shape, dtype):
        metadata = DiskTensorMetadata(shape=shape, dtype_str=str(dtype))
        return cls(path=path, metadata=metadata)

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

    def _save_safe(self):
        save_file(
            {"finalized_tensor": self.tensor},
            self.path.with_suffix(".safetensors"),
        )

    def finalize(self):
        # self.resize(self.metadata.shape[self.cat_axis], truncate=True)
        self.finalized = True
        assert not self.metadata_path.exists()
        # TODO

        self._save_safe()
        self.path.unlink()
        # assert False

        self.metadata_path.write_text(self.metadata.model_dump_json())


def main():
    path = Path("test")
    dt = DiskTensor.create(path, [10, 10], torch.int64)
    dt.tensor[:] = torch.arange(100).reshape(10, 10)
    dt.finalize()
    print(dt.tensor)

    dt = DiskTensor.open(path)
    print(dt.tensor)


if __name__ == "__main__":
    main()
