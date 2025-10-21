import asyncio
from pathlib import Path

import torch
from attrs import define, field

from saeco.data.storage.compressed_safetensors import CompressionType

from .disk_tensor import DiskTensor, DiskTensorMetadata


SAECO_MIN_GDT_INITIAL_BYTES = 2**25  # 32 MB
SAECO_MAX_GDT_INITIAL_BYTES = 2**32  # 4 GB

# class DiskTensorMetadata(DiskTensorMetadata):
#     cat_axis: int


class UnsetCatAxis:
    pass


@define
class GrowingDiskTensor(DiskTensor):
    cat_axis: int = field(init=True, default=UnsetCatAxis)
    storage_len: int | None = 2**14

    def resize(self, new_len, truncate=False):
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
            old_slice = (slice(None),) * self.cat_axis + (slice(None, new_len),)
        else:
            new_slice = (slice(None),) * self.cat_axis + (slice(None, old_len),)
            old_slice = slice(None)
        new_tensor[new_slice] = old_tensor[old_slice]
        temp.unlink()

        self.tensor = new_tensor

    @property
    def cat_len(self) -> int:
        l = self.metadata.shape[self.cat_axis]
        assert l is not None
        return l

    @property
    def valid_tensor(self):
        written_slice = [slice(None)] * self.cat_axis + [
            slice(None, self.metadata.shape[self.cat_axis])
        ]
        return self.tensor[written_slice]

    def finalize(self):
        self.resize(self.metadata.shape[self.cat_axis], truncate=True)
        super().finalize()

    def shuffle_then_finalize(
        self, shuffle_axis: int | None = None, perm: torch.Tensor | None = None
    ):
        if shuffle_axis is None:
            shuffle_axis = self.cat_axis
        self.resize(self.metadata.shape[self.cat_axis], truncate=True)
        if perm is None:
            perm = torch.randperm(self.tensor.shape[shuffle_axis])
        else:
            assert len(perm) == self.tensor.shape[shuffle_axis] and len(perm.shape) == 1
        self.tensor[:] = self.tensor.index_select(
            shuffle_axis,
            perm,
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
        compression: CompressionType = CompressionType.NONE,
        initial_nnz: int | None = None,
        cat_axis=0,
    ):
        shape = list(shape)
        if initial_nnz is None:
            # shape2 = shape.copy()
            # shape2[cat_axis] = 1
            # numel_per_nnz = torch.prod(torch.tensor(shape2)).item()
            # bytes_per_nnz = int(numel_per_nnz * dtype.itemsize)
            initial_nnz = SAECO_MIN_GDT_INITIAL_BYTES // dtype.itemsize

            # initial_nnz = 2**20
        else:
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
                shape=list(shape),
                dtype_str=str(dtype),
                compression=compression,
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
        append_slice = (slice(None),) * self.cat_axis + (
            slice(
                self.metadata.shape[self.cat_axis],
                self.metadata.shape[self.cat_axis] + length,
            ),
        )
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
