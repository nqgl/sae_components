# thanks to https://discuss.pytorch.org/t/torch-save-like-open-mode-a/137114
# for code snippets and setting me on the right path
from pathlib import Path
from typing import List, Sequence, Union

import torch
import tqdm

from saeco.data.storage.growing_disk_tensor_collection import (
    GrowingDiskTensorCollection,
)
from attrs import define
from saeco.data.piler import Piler

# path = Path("data/table_test.h5")
# t = torch.arange(32).reshape(2, 16)


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
        piler = next(iter(self.pilers.values()))
        catlens = [piler.piles.get(i).cat_len for i in range(self.num_piles)]

        perms = [torch.randperm(catlen) for catlen in catlens]
        for k, p in self.pilers.items():
            p.shuffle_piles(perms)

    def __getitem__(self, i):
        return {k: self.pilers[k][i] for k in self.keys}


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
