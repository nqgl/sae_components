from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Generic, Literal, TypeVar, cast, overload

import torch
from attrs import define

from saeco.data.storage.compressed_safetensors import CompressionType
from saeco.data.storage.disk_tensor import DiskTensor
from saeco.data.storage.growing_disk_tensor import GrowingDiskTensor
from paramsight import get_resolved_typevars_for_base, takes_alias


@define
class DiskTensorCollection[DiskTensorType: DiskTensor = DiskTensor]:
    path: Path | None = None
    stored_tensors_subdirectory_name: str = "tensors"
    return_raw: bool = False

    @property
    def disk_tensor_cls(self) -> type[DiskTensorType]:
        return self.get_disk_tensor_cls()

    @takes_alias
    @classmethod
    def get_disk_tensor_cls(cls) -> type[DiskTensorType]:
        base = get_resolved_typevars_for_base(cls, DiskTensorCollection)[0]
        assert base is not None
        return cast("type[DiskTensorType]", base)

    @property
    def storage_dir(self) -> Path:
        assert self.path is not None
        return self.path / self.stored_tensors_subdirectory_name

    def check_name_create(self, name: str | int) -> str:
        if not isinstance(name, str):
            name = str(name)
        if name in self:
            raise ValueError(f"{name} already exists!")
        return name

    def create(
        self,
        name: str,
        dtype: torch.dtype,
        shape: torch.Size | Sequence[int],
        compression: CompressionType = CompressionType.NONE,
    ) -> DiskTensorType:
        name = self.check_name_create(name)
        path = self.storage_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"Metadata already exists at {path}")

        return self.disk_tensor_cls.create(
            path=path,
            shape=tuple(shape),
            dtype=dtype,
            compression=compression,
        )

    def get(self, name: str | int) -> DiskTensorType:
        if isinstance(name, int):
            name = str(name)

        return self.disk_tensor_cls.open(self.storage_dir / name)

    def __getitem__(self, name: str | int) -> torch.Tensor | DiskTensorType:
        disk_tensor = self.get(name)
        if self.return_raw:
            return disk_tensor
        return disk_tensor.tensor

    def __setitem__(self, name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Value must be a tensor, got {type(value)}")
        disk_tensor = self.create(name, value.dtype, value.shape)
        disk_tensor.tensor[:] = value
        disk_tensor.finalize()

    def __contains__(self, name: str) -> bool:
        return name in self.keys()

    def keys(self):
        return sorted(
            {
                p.stem
                for p in self.storage_dir.glob("*")
                if p.suffix not in (".json", ".metadata")
            }
        )

    def __iter__(self):
        return iter(self.keys())

    @overload
    def items(self, raw: Literal[True] = True) -> list[tuple[str, DiskTensorType]]: ...
    @overload
    def items(self, raw: Literal[False] = False) -> list[tuple[str, torch.Tensor]]: ...
    def items(self, raw: bool = True):
        if raw:
            return [(name, self.get(name)) for name in self.keys()]
        else:
            return [(name, self.get(name).tensor) for name in self.keys()]

    @overload
    def values(self, raw: Literal[True] = True) -> list[DiskTensorType]: ...
    @overload
    def values(self, raw: Literal[False] = False) -> list[torch.Tensor]: ...
    def values(self, raw: bool = True):
        if raw:
            return [self.get(name) for name in self.keys()]
        else:
            return [self.get(name).tensor for name in self.keys()]

    def __len__(self):
        return len(self.keys())


def main():
    testdata = Path("testdata")

    # remove contents of testdata
    def rm(p):
        assert "testdata" in str(p)
        if p.is_dir():
            for child in p.iterdir():
                rm(child)
        else:
            p.unlink()

    rm(testdata)
    testdata.mkdir(parents=True, exist_ok=True)

    dtc = DiskTensorCollection[DiskTensor](
        path=testdata / "dtc1",
    )
    dtc["test"] = torch.randn(10, 10)
    print(dtc["test"])
    print(dtc.get("test").__class__)

    dt2 = DiskTensorCollection[GrowingDiskTensor](
        path=testdata / "dtc2",
    )
    dt2["test"] = torch.randn(10, 10)
    print(dt2["test"])
    print(dt2.get("test").__class__)
    print(dt2.disk_tensor_cls)
    print(dtc.disk_tensor_cls)
    print(DiskTensorCollection.disk_tensor_cls)


if __name__ == "__main__":
    main()
