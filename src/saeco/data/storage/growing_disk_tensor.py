from pathlib import Path

import torch
from attrs import define, field

from .disk_tensor import DiskTensor, DiskTensorMetadata


# class DiskTensorMetadata(DiskTensorMetadata):
#     cat_axis: int


@define
class GrowingDiskTensor(DiskTensor):
    cat_axis: int | None = None
    storage_len: int = 2**14

    def resize(self, new_len, truncate=False):
        if "9999" in self.path.name:
            print("resizing to ", new_len)
        assert not self.finalized
        old_tensor = self.tensor
        temp = self.path.rename(
            self.path.with_suffix(".old"),
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
        if "9999" in self.path.name:
            print("done resizing to ", new_len)

    @property
    def valid_tensor(self):
        written_slice = [slice(None)] * self.cat_axis + [
            slice(None, self.metadata.shape[self.cat_axis])
        ]
        return self.tensor[written_slice]

    def finalize(self):
        self.resize(self.metadata.shape[self.cat_axis], truncate=True)
        super().finalize()

    def shuffle_then_finalize(self, shuffle_axis: int | None = None):
        if shuffle_axis is None:
            shuffle_axis = self.cat_axis
        self.resize(self.metadata.shape[self.cat_axis], truncate=True)
        self.tensor[:] = self.tensor.index_select(
            shuffle_axis, torch.randperm(self.tensor.shape[shuffle_axis])
        )
        super().finalize()

    @classmethod
    def open(cls, path: Path):
        inst = cls(
            path=path,
            metadata=cls._open_metadata(path),
            storage_len=None,
        )
        assert inst.finalized
        return inst

    @classmethod
    def create(
        cls,
        path: Path,
        shape: list[int],
        dtype: torch.dtype,
        initial_nnz: int | None = None,
        cat_axis=0,
    ):
        if initial_nnz is None:
            initial_nnz = 2**15
        else:
            shape = shape.copy()
            shape[cat_axis] = None
        cat_len = shape[cat_axis] or int(
            initial_nnz
            // torch.prod(torch.tensor(shape[:cat_axis] + shape[cat_axis + 1 :])).item()
            + 1
        )
        assert cat_len > 0
        shape[cat_axis] = 0
        inst = cls(
            path=path,
            storage_len=cat_len,
            cat_axis=cat_axis,
            metadata=DiskTensorMetadata(
                shape=shape,
                dtype_str=str(dtype),
            ),
            finalized=False,
        )
        return inst

    @property
    def storage_shape(self):
        return [
            s if i != self.cat_axis else self.storage_len
            for i, s in enumerate(self.metadata.shape)
        ]

    def append(self, tensor):
        length = tensor.shape[self.cat_axis]
        assert (
            list(tensor.shape[: self.cat_axis]) == self.metadata.shape[: self.cat_axis]
            and list(tensor.shape[self.cat_axis + 1 :])
            == self.metadata.shape[self.cat_axis + 1 :]
        )
        assert not self.finalized
        while self.metadata.shape[self.cat_axis] + length >= self.storage_len:
            self.resize(self.storage_len * 2)
        append_slice = [slice(None)] * self.cat_axis + [
            slice(
                self.metadata.shape[self.cat_axis],
                self.metadata.shape[self.cat_axis] + length,
            )
        ]
        self.tensor[append_slice] = tensor
        self.metadata.shape[self.cat_axis] += length


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
