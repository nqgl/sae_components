from attr import define, field
from pathlib import Path
import torch

from .disk_tensor import DiskTensorMetadata


@define
class GrowingDiskTensor:
    path: Path
    shape: list[int]
    dtype: torch.dtype
    cat_axis: int = 0
    finalized: bool = False
    variable_axis_len: int = 0
    storage_len: int = 2**14
    tensor: torch.Tensor = field(init=False)

    @tensor.default
    def _tensor_default(self):
        if self.path.exists():
            self.finalized = True
        return self.create_tensor()

    def create_tensor(self):
        return self.tensorclass(
            torch.UntypedStorage.from_file(
                str(self.path), shared=True, nbytes=self.nbytes(self.storage_len)
            )
        ).reshape(
            *[
                s if i != self.cat_axis else self.storage_len
                for i, s in enumerate(self.shape)
            ]
        )

    def path_variation(self, s):
        return self.path.with_name(self.path.name + str(s))

    @property
    def tensorclass(self):
        if self.dtype == torch.float32:
            return torch.FloatTensor
        elif self.dtype == torch.int64:
            return torch.LongTensor
        elif self.dtype == torch.bool:
            return torch.BoolTensor
        elif self.dtype == torch.float16:
            return torch.HalfTensor
        else:
            raise ValueError(f"Unsupported dtype {self.dtype}")

    def resize(self, new_len, truncate=False):
        assert not self.finalized
        old_tensor = self.tensor
        temp = self.path.rename(
            self.path_variation("old"),
        )
        old_len = self.storage_len
        self.storage_len = new_len
        new_tensor = self.create_tensor()
        if truncate:
            new_slice = slice(None)
            old_slice = [slice(None)] * self.cat_axis + [slice(None, new_len)]
        else:
            new_slice = [slice(None)] * self.cat_axis + [slice(None, old_len)]
            old_slice = slice(None)
        new_tensor[new_slice] = old_tensor[old_slice]
        temp.unlink()

        self.tensor = new_tensor

    def finalize(self):
        self.resize(self.variable_axis_len, truncate=True)
        self.shape = list(self.shape)
        self.shape[self.cat_axis] = self.variable_axis_len
        self.finalized = True
        metadata = DiskTensorMetadata(
            shape=self.shape,
            dtype=str(self.dtype),
            length=self.variable_axis_len,
            cat_axis=self.cat_axis,
        )
        metadata_path = self.path.with_suffix(".metadata")
        assert not metadata_path.exists()
        metadata_path.write_text(metadata.model_dump_json())

    @classmethod
    def open(cls, path: Path):
        metadata = DiskTensorMetadata.model_validate_json(
            (path.with_suffix(".metadata")).read_text()
        )
        inst = cls(
            path=path,
            shape=metadata.shape,
            dtype=metadata.torch_dtype,
            storage_len=metadata.length,
            cat_axis=metadata.cat_axis,
        )
        assert inst.finalized
        return inst

    @classmethod
    def create(
        cls,
        path: Path,
        shape: list[int],
        dtype: torch.dtype,
        initial_nnz: int = 4096,
        cat_axis=0,
    ):
        assert initial_nnz > 0
        inst = cls(
            path=path,
            shape=shape,
            storage_len=initial_nnz,
            dtype=dtype,
            cat_axis=cat_axis,
        )
        return inst

    def nbytes(self, length):
        size = self.dtype.itemsize
        for i, s in enumerate(self.shape):
            if i == self.cat_axis:
                size *= length
            else:
                size *= s
        return size

    def append(self, tensor):
        length = tensor.shape[self.cat_axis]
        assert (
            list(tensor.shape[: self.cat_axis]) == self.shape[: self.cat_axis]
            and list(tensor.shape[self.cat_axis + 1 :])
            == self.shape[self.cat_axis + 1 :]
        )
        assert not self.finalized
        while self.variable_axis_len + length >= self.storage_len:
            self.resize(self.storage_len * 2)
        append_slice = [slice(None)] * self.cat_axis + [
            slice(self.variable_axis_len, self.variable_axis_len + length)
        ]
        self.tensor[append_slice] = tensor
        self.variable_axis_len += length


if __name__ == "__main__":
    path = Path("test2.bin")
    shape = [128, 128, 512]
    dtype = torch.float32
    tensor = torch.randn(*shape, dtype=dtype)
    if not path.exists():
        growing_disk_tensor = GrowingDiskTensor.create(
            path, shape, dtype, initial_nnz=1, cat_axis=1
        )
        for i in range(50):
            sh = shape.copy()
            sh[1] = 111
            t = (
                torch.arange(i * 111, i * 111 + 111, dtype=dtype)
                .unsqueeze(-1)
                .unsqueeze(0)
                .expand(*[sh])
            )
            growing_disk_tensor.append(t)
        growing_disk_tensor.finalize()
    else:
        growing_disk_tensor = GrowingDiskTensor.open(path)
    print(growing_disk_tensor.tensor[:, 4])
    print(growing_disk_tensor.tensor[:, 4444])
    print(growing_disk_tensor.tensor[4])
    # print(growing_disk_tensor.tensor[4444])

    print(growing_disk_tensor.tensor.shape)
    print(growing_disk_tensor.tensor.dtype)
