import asyncio
from pathlib import Path
from typing import Sequence

import torch
import tqdm
from attrs import define, field
from pydantic import BaseModel

from saeco.data.storage.compressed_safetensors import CompressionType

from saeco.data.storage.growing_disk_tensor import GrowingDiskTensor

from . import DiskTensor, DiskTensorCollection, GrowingDiskTensor


class GrowingDiskTensorCollectionMetadata(BaseModel):
    finalized: bool = False

    def save(self, path: Path) -> None:
        self.metadata_path_from_dir(path).write_text(self.model_dump_json())

    @classmethod
    def metadata_path_from_dir(cls, path: Path) -> Path:
        assert path.is_dir()
        return path / "collection_metadata.json"

    @classmethod
    def load_from_dir(
        cls, path: Path, assert_exists: bool = False
    ) -> "GrowingDiskTensorCollectionMetadata":
        path.mkdir(parents=True, exist_ok=True)
        metadata_path = cls.metadata_path_from_dir(path)
        if not metadata_path.exists():
            if assert_exists:
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            return cls()
        return cls.model_validate_json(metadata_path.read_text())


from attrs import Factory


def _metadata_default(
    self: "GrowingDiskTensorCollection",
) -> GrowingDiskTensorCollectionMetadata:
    return GrowingDiskTensorCollectionMetadata.load_from_dir(
        self.storage_dir, assert_exists=False
    )


@define
class GrowingDiskTensorCollection(DiskTensorCollection[GrowingDiskTensor]):
    metadata: GrowingDiskTensorCollectionMetadata = field(
        default=Factory(_metadata_default, takes_self=True)
    )
    cache: dict[str, GrowingDiskTensor] = field(factory=dict)
    skip_cache: bool = False

    def get(self, name: str | int) -> GrowingDiskTensor:
        if isinstance(name, int):
            name = str(name)
        assert isinstance(name, str)
        if self.skip_cache:
            return super().get(name)
        if name not in self.cache:
            self.cache[name] = super().get(name)
        return self.cache[name]

    @property
    def finalized(self) -> bool:
        return self.metadata.finalized

    def __setitem__(self, name: str, value: torch.Tensor) -> None:
        raise ValueError("Cannot set items in a growing collection")

    def create(
        self,
        name: str | int,
        dtype: torch.dtype,
        shape: torch.Size | Sequence[int],
        compression: CompressionType = CompressionType.NONE,
    ) -> GrowingDiskTensor:
        if self.finalized:
            raise ValueError("Collection is finalized, cannot create new tensors")
        name = self.check_name_create(name)
        self.cache[name] = super().create(name, dtype, shape, compression=compression)
        return self.cache[name]

    def keys(self):
        return sorted(list(set(self.cache.keys()) | set(super().keys())))

    def shuffle_then_finalize(self, perms: dict[str, torch.Tensor] | None = None):
        if any([(f := self.get(name)).finalized for name in self.keys()]):
            raise ValueError(
                f"Cannot shuffle finalized tensors: tensor {f} is finalized"
            )
        tkeys = tqdm.tqdm(self.keys())
        tkeys.set_description("Shuffling")
        if perms is not None:
            assert len(perms) == len(self.keys())
        for name in tkeys:
            dt = self.get(name)
            dt.shuffle_then_finalize(perm=perms[name] if perms is not None else None)
        self.metadata.finalized = True
        self.metadata.save(self.storage_dir)

    async def _async_shuffle_then_finalize(
        self, perms: dict[str, torch.Tensor] | None = None
    ):
        if any([(f := self.get(name)).finalized for name in self.keys()]):
            raise ValueError(
                f"Cannot shuffle finalized tensors: tensor {f} is finalized"
            )
        tkeys = tqdm.tqdm(self.keys())
        tkeys.set_description("Shuffling")
        if perms is not None:
            assert len(perms) == len(self.keys())

        await asyncio.gather(
            *[
                asyncio.to_thread(
                    self.get(name).shuffle_then_finalize,
                    perm=perms[name] if perms is not None else None,
                )
                for name in self.keys()
            ]
        )
        self.metadata.finalized = True
        self.metadata.save(self.storage_dir)

    def shuffle_then_finalize_async(self, perms: dict[str, torch.Tensor] | None = None):
        asyncio.run(self._async_shuffle_then_finalize(perms))

    def finalize(self) -> None:
        tkeys = tqdm.tqdm(self.keys())
        tkeys.set_description("Shuffling")
        for name in tkeys:
            dt = self.get(name)
            if not dt.finalized:
                dt.finalize()
        self.metadata.finalized = True
        self.metadata.save(self.storage_dir)

    def __contains__(self, name: str) -> bool:
        if isinstance(name, int):
            name = str(name)
        return super().__contains__(name)
