# thanks to https://discuss.pytorch.org/t/torch-save-like-open-mode-a/137114
# for code snippets and setting me on the right path
from pathlib import Path
from typing import List, Sequence, Union, cast

import torch
import tqdm

from saeco.data.storage.growing_disk_tensor_collection import (
    GrowingDiskTensorCollection,
)
from attrs import define
from saeco.data.piler import Piler

# path = Path("data/table_test.h5")
# t = torch.arange(32).reshape(2, 16)

from typing import overload


@define
class DictBatch:
    data: dict[str, torch.Tensor]

    @overload
    def __getitem__(self, i: str) -> torch.Tensor: ...
    @overload
    def __getitem__(self, i: int | slice | list[int] | torch.Tensor) -> "DictBatch": ...
    def __getitem__(self, i):
        if isinstance(i, str):
            return self.data[i]
        return DictBatch({k: v[i] for k, v in self.data.items()})

    def to(self, device):
        return DictBatch({k: v.to(device) for k, v in self.data.items()})

    def cuda(self):
        return self.to("cuda")

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def cat(self, other: "DictBatch"):
        assert self.keys() == other.keys()
        return DictBatch(
            {
                k: torch.cat([self.data[k], other.data[k]], dim=0)
                for k in self.data.keys()
            }
        )

    def __len__(self):
        l = len(next(iter(self.data.values())))
        assert all(len(v) == l for v in self.data.values())
        return l

    @classmethod
    def cat_list(cls, batches: list["DictBatch"], dim: int = 0):
        keys = batches[0].data.keys()
        assert all(b.data.keys() == keys for b in batches)
        return DictBatch(
            {k: torch.cat([b.data[k] for b in batches], dim=dim) for k in keys}
        )


class DictPiler:
    def __init__(
        self,
        path: Union[str, Path],
        dtypes: dict[str, torch.dtype],
        fixed_shapes: dict[str, torch.Size] | dict[str, Sequence[int]],
        num_piles=None,
    ):
        self.keys = set(dtypes.keys())

        assert fixed_shapes is None or self.keys == set(fixed_shapes.keys())
        if isinstance(path, str):
            path = Path(path)

        shapes = {k: [0] + list(v) for k, v in fixed_shapes.items()}
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

        self.dtypes = dtypes
        self.shapes = shapes
        if not self.path.exists():
            self.path.mkdir(parents=True)
        self.pilers = {
            k: Piler(
                self.path / k, dtype=dtypes[k], shape=shapes[k], num_piles=num_piles
            )
            for k in self.keys
        }
        if num_piles is None:
            assert all(
                p.num_piles == self.pilers[next(iter(self.keys))].num_piles
                for p in self.pilers.values()
            )
            self.num_piles = self.pilers[next(iter(self.keys))].num_piles
        else:
            self.num_piles = num_piles

    def distribute(
        self, tensors: dict[str, torch.Tensor], indexer: torch.Tensor | None = None
    ):
        if indexer is None:
            i = torch.randint(
                0, self.num_piles, [tensors[next(iter(self.keys))].shape[0]]
            )
        else:
            i = indexer
        assert all(t.shape[0] == i.shape[0] for t in tensors.values())
        assert self.keys == set(tensors.keys())
        for k, t in tensors.items():
            self.pilers[k].distribute(t, i)
        return i

    def shuffle_piles(self):
        # tqdm.tqdm.write("Shuffling piles")
        if self.readonly:
            raise ValueError("Cannot write to a readonly Piler")
        i = iter(self.pilers.values())
        piler = next(i)
        catlens = {k: piler.piles.get(k).cat_len for k in piler.piles.keys()}

        perms = {k: torch.randperm(catlen) for k, catlen in catlens.items()}
        for k, p in self.pilers.items():
            p.shuffle_piles(perms)

    def __getitem__(self, i) -> DictBatch:
        data = {k: self.pilers[k][i] for k in self.keys}
        assert isinstance(data, dict)
        assert all(isinstance(v, torch.Tensor) for v in data.values())
        return DictBatch(data=cast(dict[str, torch.Tensor], data))

    def batch_generator(
        self,
        batch_size,
        id=None,
        nw=None,
    ):
        assert id == nw == None or id is not None and nw is not None
        id = id or 0
        nw = nw or 1

        spares: list[DictBatch] = []
        nspare = 0
        for p in range(id % nw, self.num_piles, nw):
            pile = self[p]
            for i in range(0, len(pile) // batch_size * batch_size, batch_size):
                yield pile[i : i + batch_size]
            spare = pile[len(pile) // batch_size * batch_size :]
            if len(spare) > 0:
                spares.append(spare)
                nspare += len(spare)
                if nspare > batch_size:
                    consolidated = DictBatch.cat_list(spares, dim=0)
                    for i in range(
                        0, len(consolidated) // batch_size * batch_size, batch_size
                    ):
                        yield consolidated[i : i + batch_size]
                    spare = consolidated[len(consolidated) // batch_size * batch_size :]
                    spares = [spare]
                    nspare = len(spare)

    def as_dataset(self, batch_size, converter=None):
        return PilerDataset(self, batch_size, converter)


import torch.utils.data


class PilerDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset for activations
    """

    def __init__(self, piler: DictPiler, batch_size, converter=None):
        self.batch_size = batch_size
        self.piler = piler
        self.converter = converter

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self.piler.batch_generator(self.batch_size)

        else:
            id = worker_info.id
            nw = worker_info.num_workers
            assert id % nw == id, (id, nw)

            gen = self.piler.batch_generator(
                batch_size=self.batch_size,
                id=id,
                nw=nw,
            )
        if self.converter is not None:
            return self.converter(gen)
        return gen


def main():
    test_path = Path("testdata")
    dp = DictPiler(
        test_path,
        {"a": torch.int64, "b": torch.int64},
        fixed_shapes={"a": [16], "b": [16, 16]},
        num_piles=10,
    )

    for i in range(1000):
        t1 = torch.arange(i, i + 16)
        t2 = torch.arange(i, i + 16).unsqueeze(1) + t1
        dp.distribute({"a": t1.unsqueeze(0), "b": t2.unsqueeze(0)})

    dp.shuffle_piles()

    print(dp[0])
    dp[0:2]
    dp[0:2]
    dd = dp[8]
    a = dd["a"]
    b = dd["b"]
    a.shape
    b.shape
    i = 10
    a[i]
    b[i] // 2
    a[i]
    aa = a[i].unsqueeze(1) + a[i]
    assert (aa - b[i] == 0).all()
    a[0]


if __name__ == "__main__":
    main()
