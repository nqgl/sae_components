# thanks to https://discuss.pytorch.org/t/torch-save-like-open-mode-a/137114
# for code snippets and setting me on the right path
from pathlib import Path
from typing import List, Union

import tables
import torch
import tqdm

from saeco.data.piler.append_disktensor import AppendDiskTensor

# path = Path("data/table_test.h5")
# t = torch.arange(32).reshape(2, 16)


class Piler:
    def __init__(self, path: Union[str, Path], dtype, fixed_shape, num_piles=None):
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.readonly = num_piles is None
        if num_piles is None:
            g = path.glob("pile*")
            num_piles = len(list(g))
            assert num_piles > 0
        else:
            path.mkdir(parents=True)
            g = path.glob("pile*")
            assert len(list(g)) == 0

        self.dtype = dtype
        self.fixed_shape = fixed_shape
        self.num_piles = num_piles
        self.piles: List[AppendDiskTensor] = [
            AppendDiskTensor(path / f"pile_{i}.h5", dtype, fixed_shape)
            for i in range(num_piles)
        ]

    def distribute(self, t: torch.Tensor):
        if self.readonly:
            raise ValueError("Cannot write to a readonly Piler")
        i = torch.randint(0, self.num_piles, [t.shape[0]])
        for pile in range(self.num_piles):
            self.piles[pile].write(t[i == pile])

    def shuffle_piles(self):
        tqdm.tqdm.write("Shuffling piles")
        if self.readonly:
            raise ValueError("Cannot write to a readonly Piler")

        for pile in tqdm.tqdm(self.piles):
            pile.shuffle_and_finalize()

    def __getitem__(self, i):
        if isinstance(i, int):
            piles = [self.piles[i]]
        elif isinstance(i, list):
            piles = [self.piles[j] for j in i]
        else:
            piles = self.piles[i]
        if len(piles) == 1:
            return piles[0].readtensor()
        return torch.cat([p.readtensor() for p in piles])


def main():

    t_app = AppendDiskTensor(
        "data/table_test.h5",
        torch.int64,
        [16],
    )
    p = Piler("data/piler_test", torch.int64, [16], num_piles=4)
    for i in range(400):
        t = torch.arange(32000).reshape(-1, 16)
        p.distribute(t)
        print()
    p.shuffle_piles()


if __name__ == "__main__":
    main()
