from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.data.storage.disk_tensor import DiskTensor


import torch
from attrs import define


from pathlib import Path


@define
class DiskTensorCollection:
    path: Path
    stored_tensors_subdirectory_name: str = "tensors"
    return_raw: bool = False

    @property
    def storage_dir(self) -> Path:
        return self.path / self.stored_tensors_subdirectory_name

    def create(self, name: str, dtype: torch.dtype, shape: torch.Size) -> DiskTensor:
        path = self.storage_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"Metadata already exists at {path}")
        return DiskTensor.create(
            path=path,
            shape=shape,
            dtype=dtype,
        )

    def get(self, name):
        return DiskTensor.open(self.storage_dir / name)

    def __getitem__(self, name):
        if self.return_raw:
            return DiskTensor.open(self.storage_dir / name)
        return DiskTensor.open(self.storage_dir / name).tensor

    def __setitem__(self, name, value):
        assert isinstance(name, str)
        assert isinstance(value, torch.Tensor)
        assert not name in self
        disk_tensor = self.create(name, value.dtype, value.shape)
        disk_tensor.tensor[:] = value
        disk_tensor.finalize()

    def __contains__(self, name):
        return name in self.keys()

    def keys(self):
        return list(set([p.stem for p in self.storage_dir.glob("*")]))
