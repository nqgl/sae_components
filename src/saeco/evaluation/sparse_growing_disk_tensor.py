from saeco.evaluation.growing_disk_tensor import GrowingDiskTensor


import torch
from attr import define, field


from pathlib import Path
from pydantic import BaseModel


class SparseGrowingDiskTensorMetadata(BaseModel):
    shape: list[int]
    dtype: str
    cat_axis: int

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


@define
class SparseGrowingDiskTensor:
    path: Path
    shape: list[int]
    dtype: torch.dtype = torch.float32
    indices: GrowingDiskTensor = field(init=False)
    values: GrowingDiskTensor = field(init=False)
    cat_axis: int = 0
    finalized: bool = False

    @indices.default
    def _indices_default(self):
        if self.indices_path.exists():
            return GrowingDiskTensor.open(path=self.indices_path)
        return GrowingDiskTensor.create(
            path=self.indices_path,
            shape=[len(self.shape), None],
            cat_axis=1,
            dtype=torch.int64,
        )

    @values.default
    def _values_default(self):
        if self.values_path.exists():
            return GrowingDiskTensor.open(path=self.values_path)
        return GrowingDiskTensor.create(
            path=self.values_path, shape=[None], cat_axis=0, dtype=self.dtype
        )

    def append(self, tensor: torch.Tensor):
        if not tensor.is_sparse:
            tensor = tensor.to_sparse_coo()
        if not tensor.is_coalesced():
            tensor = tensor.coalesce()

        indices = tensor.indices()
        indices[self.cat_axis] += self.shape[
            self.cat_axis
        ]  # shift indices to this space
        self.shape[self.cat_axis] += tensor.shape[self.cat_axis]
        if indices.numel() == 0:
            return
        for i, idx in enumerate(self.shape):
            assert indices[i].max() < idx and indices[i].min() >= 0
        self.indices.append(indices)
        self.values.append(tensor.values())
        assert self.values.variable_axis_len == self.indices.variable_axis_len

    def finalize(self):
        self.indices.finalize()
        self.values.finalize()
        metadata = SparseGrowingDiskTensorMetadata(
            dtype=str(self.dtype), shape=self.shape, cat_axis=self.cat_axis
        )
        (self.path / "metadata.json").write_text(metadata.model_dump_json())

    @property
    def tensor(self):
        assert self.indices.finalized and self.values.finalized
        return torch.sparse_coo_tensor(
            self.indices.tensor, self.values.tensor, self.shape
        ).coalesce()

    @property
    def indices_path(self):
        return self.path / "indices.bin"

    @property
    def values_path(self):
        return self.path / "values.bin"

    @classmethod
    def create(cls, path: Path, shape: list[int], cat_axis=0, dtype=torch.float32):
        assert not path.exists()
        path.mkdir(parents=True)
        shape = shape.copy()
        shape[cat_axis] = 0
        inst = cls(path=path, shape=shape, cat_axis=cat_axis, dtype=dtype)
        return inst

    @classmethod
    def open(cls, path: Path):
        metadata = SparseGrowingDiskTensorMetadata.model_validate_json(
            (path / "metadata.json").read_text()
        )
        inst = cls(
            path=path,
            shape=metadata.shape,
            dtype=metadata.torch_dtype,
            cat_axis=metadata.cat_axis,
        )
        return inst


if __name__ == "__main__":
    path = Path("test_sparse")
    # sgdt = SparseGrowingDiskTensor.create(path=path, shape=[0, 10, 10])
    # for i in range(100):
    #     t = torch.arange(1000 * i, 1000 * (i + 1)).reshape(-1, 10, 10)
    #     sgdt.append(t)
    sgdt = SparseGrowingDiskTensor.open(path)
    # sgdt.finalize()
    print(sgdt.tensor)
