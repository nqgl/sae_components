# thanks to https://discuss.pytorch.org/t/torch-save-like-open-mode-a/137114
# for code snippets and setting me on the right path
from pathlib import Path
from typing import List, Union

import torch
import tqdm
from attrs import define
import asyncio
from saeco.data.storage.growing_disk_tensor_collection import (
    GrowingDiskTensorCollection,
)


# path = Path("data/table_test.h5")
# t = torch.arange(32).reshape(2, 16)


class Piler:
    def __init__(
        self,
        path: Union[str, Path],
        dtype,
        fixed_shape: list[int] | None = None,
        shape: list[int] | None = None,
        num_piles: int | None = None,
    ):
        if isinstance(path, str):
            path = Path(path)
        if shape is None:
            assert fixed_shape is not None
            shape = [0] + list(fixed_shape)
        else:
            assert fixed_shape is None
        self.path = path
        self.readonly = num_piles is None
        # if num_piles is None:
        #     g = path.glob("pile*")
        #     num_piles = len(list(g))
        #     assert num_piles > 0
        # else:
        #     path.mkdir(parents=True)
        #     g = path.glob("pile*")
        #     assert len(list(g)) == 0

        self.dtype = dtype
        self.shape = shape
        if not self.path.exists():
            self.path.mkdir(parents=True)
        self.piles = GrowingDiskTensorCollection(
            self.path, stored_tensors_subdirectory_name="piles"
        )
        if num_piles is None:
            self.num_piles = len(self.piles)
        else:
            self.num_piles = num_piles
        if not self.readonly and len(self.piles) == 0:
            self.initialize()

    def initialize(self):
        for i in range(self.num_piles):
            self.piles.create(i, self.dtype, torch.Size(self.shape))

    # def finalize(self):
    #     self.piles.finalize()
    #     assert len(self.piles) == self.num_piles

    def distribute(self, t: torch.Tensor, indexer: torch.Tensor | None = None):
        if self.readonly:
            raise ValueError("Cannot write to a readonly Piler")
        if t.dtype != self.dtype:
            raise ValueError(
                f"Tensor dtype {t.dtype} does not match Piler dtype {self.dtype}"
            )
        if indexer is None:
            i = torch.randint(0, self.num_piles, [t.shape[0]])
        else:
            i = indexer
        trng = tqdm.trange(self.num_piles, leave=False)
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
        if t.dtype != self.dtype:
            raise ValueError(
                f"Tensor dtype {t.dtype} does not match Piler dtype {self.dtype}"
            )
        # Compute indexer tensor if not provided
        if indexer is None:
            i = torch.randint(0, self.num_piles, [t.shape[0]])
        else:
            i = indexer

        # Create an async wrapper for a single pile append.
        async def append_to_pile(pile_idx: int):
            # Get the corresponding pile and data for it.
            pile = self.piles.get(pile_idx)
            data_to_append = t[i == pile_idx]
            # Offload the blocking append call to a worker thread.
            await asyncio.to_thread(pile.append, data_to_append)

        # Use tqdm for progress - note that tqdm is not asynchronous but
        # we can still update it as long as the loop scheduling is done in the main thread.
        # Prepare tasks and attach a simple progress counter.
        tasks = []
        for pile_idx in range(self.num_piles):
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

    def __getitem__(self, i):
        if isinstance(i, int):
            piles = [self.piles[i]]
        elif isinstance(i, list):
            piles = [self.piles[j] for j in i]
        elif isinstance(i, slice):
            piles = [self.piles[j] for j in self.piles.keys()[i]]
        else:
            piles = self.piles[i]
        if isinstance(piles, list) and len(piles) == 1:
            return piles[0]
        return torch.cat(piles)  # type: ignore


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

    p = Piler(testdata / "piler_test", torch.int64, [None, 16], num_piles=4)
    for i in range(400):
        t = torch.arange(32000).reshape(-1, 16)
        p.distribute(t)
        print()
    unshuffled = torch.cat([dt.valid_tensor for dt in p.piles.values()])

    p.shuffle_piles()
    shuffled = torch.cat(p.piles.values(raw=False))
    p2 = Piler(testdata / "piler_test", torch.int64, [None, 16])
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
