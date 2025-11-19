# thanks to https://discuss.pytorch.org/t/torch-save-like-open-mode-a/137114
# for code snippets and setting me on the right path
import asyncio
import os
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

import torch
import tqdm
from attrs import define
from pydantic import BaseModel

from saeco.data.storage.compressed_safetensors import CompressionType
from saeco.data.storage.growing_disk_tensor_collection import (
    GrowingDiskTensorCollection,
)
from saeco.misc import str_to_dtype


class PilerMetadata(BaseModel):
    dtype: str
    fixed_shape: list[int]
    compression: CompressionType = CompressionType.NONE
    num_piles: int


T = TypeVar("T")


def assert_cast(tp: type[T], value: Any) -> T:
    if not isinstance(value, tp):
        raise TypeError(f"Expected {tp.__name__}, got {type(value).__name__}")
    return value


@define
class Piler:
    metadata: PilerMetadata
    path: Path
    readonly: bool
    piles: GrowingDiskTensorCollection

    @classmethod
    def create(
        cls,
        path: str | Path,
        dtype: torch.dtype | str,
        fixed_shape: list[int],
        num_piles: int,
        compress: bool = False,
    ):
        if isinstance(path, str):
            path = Path(path)

        if isinstance(dtype, torch.dtype):
            dtype = str(dtype)

        compression = CompressionType.NONE if not compress else CompressionType.ZSTD

        metadata = PilerMetadata(
            dtype=dtype,
            fixed_shape=fixed_shape,
            num_piles=num_piles,
            compression=compression,
        )

        if path.exists():
            raise ValueError(f"folder already exists at {path}")
        path.mkdir(parents=True)

        piles = GrowingDiskTensorCollection(
            path, stored_tensors_subdirectory_name="piles"
        )

        for i in range(num_piles):
            piles.create(
                i, dtype=dtype, shape=[0] + list(fixed_shape), compression=compression
            )

        piler = Piler(
            metadata=metadata,
            path=path,
            readonly=False,
            piles=piles,
        )

        cls.get_metadata_path(path).write_text(metadata.model_dump_json())

        return piler

    @classmethod
    def open(cls, path: str | Path, skip_cache: bool = False):
        if isinstance(path, str):
            path = Path(path)
        metadata_path = cls.get_metadata_path(path)

        if not metadata_path.exists():
            raise ValueError(f"Piler metadata not found at {metadata_path}")

        metadata = PilerMetadata.model_validate_json(metadata_path.read_text())

        gdtc = GrowingDiskTensorCollection(
            path,
            stored_tensors_subdirectory_name="piles",
            skip_cache=skip_cache,
        )

        if len(gdtc) != metadata.num_piles:
            raise ValueError(
                f"expected {metadata.num_piles} piles, got {len(gdtc)} for {path}"
            )

        piler = Piler(metadata, path, readonly=True, piles=gdtc)

        return piler

    @property
    def num_piles(self):
        return self.metadata.num_piles

    @property
    def dtype(self) -> torch.dtype:
        return str_to_dtype(self.metadata.dtype)

    @classmethod
    def get_metadata_path(cls, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        return path / "piler_metadata.json"

    def distribute(self, t: torch.Tensor, indexer: torch.Tensor | None = None):
        if self.readonly:
            raise ValueError("Cannot write to a readonly Piler")
        if str(t.dtype) != self.metadata.dtype:
            raise ValueError(
                f"Tensor dtype {t.dtype} does not match Piler dtype {self.metadata.dtype}"
            )
        if indexer is None:
            i = torch.randint(0, self.metadata.num_piles, [t.shape[0]])
        else:
            i = indexer
        trng = tqdm.trange(self.metadata.num_piles, leave=False)
        trng.set_description(f"Distributing {t.shape[0]}")
        for pile_idx in trng:
            self.piles.get(pile_idx).append(t[i == pile_idx])
        return i

    def distribute_async(self, t: torch.Tensor, indexer: torch.Tensor | None = None):
        return asyncio.run(self._async_distribute(t, indexer))

    async def _async_distribute(
        self, t: torch.Tensor, indexer: torch.Tensor | None = None
    ):
        if self.readonly:
            raise ValueError("Cannot write to a readonly Piler")
        if str(t.dtype) != self.metadata.dtype:
            raise ValueError(
                f"Tensor dtype {t.dtype} does not match Piler dtype {self.metadata.dtype}"
            )
        # Compute indexer tensor if not provided
        if indexer is None:
            i = torch.randint(0, self.metadata.num_piles, [t.shape[0]])
        else:
            i = indexer

        # Create an async wrapper for a single pile append.
        sem = asyncio.Semaphore(
            int(os.environ.get("SAECO_ASYNC_DISTRIBUTE_WORKERS", 4096))
        )

        async def append_to_pile(pile_idx: int):
            async with sem:
                # Get the corresponding pile and data for it.
                pile = self.piles.get(pile_idx)
                data_to_append = t[i == pile_idx]
                # Offload the blocking append call to a worker thread.
                await asyncio.to_thread(pile.append, data_to_append)

        # Use tqdm for progress - note that tqdm is not asynchronous but
        # we can still update it as long as the loop scheduling is done in the main thread.
        # Prepare tasks and attach a simple progress counter.
        tasks = []
        for pile_idx in range(self.metadata.num_piles):
            tasks.append(asyncio.create_task(append_to_pile(pile_idx)))

        # Optionally, you can monitor task completion via tqdm:
        for f in tqdm.tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Distributing {t.shape[0]}",
        ):
            await f

        return i

    def shuffle_piles(self, perms: dict[str, torch.Tensor] | None = None):
        # tqdm.tqdm.write("Shuffling piles")
        if self.readonly:
            raise ValueError("Cannot write to a readonly Piler")
        self.piles.shuffle_then_finalize(perms=perms)

    def __getitem__(self, i) -> torch.Tensor:
        if isinstance(i, int):
            piles = [self.piles[i]]
        elif isinstance(i, list):
            piles = [self.piles[j] for j in i]
        elif isinstance(i, slice):
            piles = [self.piles[j] for j in self.piles.keys()[i]]
        else:
            piles = self.piles[i]
        if isinstance(piles, list) and len(piles) == 1:
            assert isinstance(piles[0], torch.Tensor)
            return piles[0]
        return torch.cat(piles)  # type: ignore

    @cached_property
    def shapes(self) -> list[list[int]]:
        assert self.piles.finalized
        return [
            [assert_cast(int, i) for i in self.piles.get(n).metadata.shape]
            for n in range(self.num_piles)
        ]

    @cached_property
    def shape(self) -> list[int]:
        batch = sum([shape[0] for shape in self.shapes])
        rest = self.shapes[0][1:]
        assert all(shape[1:] == rest for shape in self.shapes)
        return [batch] + rest

    @property
    def num_samples(self) -> int:
        return self.shape[0]


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

    p = Piler.create(testdata / "piler_test", torch.int64, [16], 4)
    for i in range(400):
        t = torch.arange(32000).reshape(-1, 16)
        p.distribute(t)
        print()
    unshuffled = torch.cat([dt.valid_tensor for dt in p.piles.values()])

    p.shuffle_piles()
    shuffled = torch.cat(p.piles.values(raw=False))

    p2 = Piler.open(testdata / "piler_test")
    reopened = torch.cat(p2.piles.values(raw=False))
    p.piles[0] == p2.piles[0]
    p.piles.keys()
    p2.piles.keys()
    assert torch.allclose(shuffled, reopened)
    assert not torch.allclose(unshuffled, shuffled)
    d = {}
    assert (shuffled.sum(1) == reopened.sum(1)).all()
    for i in shuffled.flatten().tolist():
        # assert i not in d
        if i not in d:
            d[i] = 0
        d[i] += 1
    u, n = reopened.unique(return_counts=True)

    u.shape
    (n == 400).all()
    for i in reopened.flatten().tolist():
        d[i] += 10
    for i in unshuffled.flatten().tolist():
        d[i] += 100
    assert len(d) == 32000
    for i in range(32000):
        assert d[i] == 111 * 400
    print("done")
    unshuffled.shape


if __name__ == "__main__":
    main()
