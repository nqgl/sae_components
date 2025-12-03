import itertools
import random
from collections.abc import Callable, Generator, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import (
    Literal,
    cast,
    overload,
)

import torch
import torch.utils.data
from attrs import define
from pydantic import BaseModel

from saeco.data.dict_batch.dict_batch import DictBatch
from saeco.data.piler import Piler


class DictPilerMetadata(BaseModel):
    keys: set[str]


def shuffled_range(start, stop, mod, shuffle=True):
    shrange = list(range(start, stop, mod))
    if not shuffle:
        return shrange
    random.shuffle(shrange)
    return shrange


### TODO flip:
# currently it's DictPiler[tensor name key -> Piler[ index -> Pile]]
# seems better to flip it to DictPiler[ index -> DictPile[tensor name key -> Pile/tensor]]
# not high priority though


@define
class DictPiler:
    path: Path
    use_async_distribute: bool = (
        True  # this is a flag just bc it's a new/experimental feature
    )
    readonly: bool = False

    @classmethod
    def create(
        cls,
        path: str | Path,
        dtypes: Mapping[str, torch.dtype] | Mapping[str, str],
        fixed_shapes: Mapping[str, torch.Size] | Mapping[str, Sequence[int]],
        num_piles: int,
        use_async_distribute: bool = True,
        compress: bool = False,
    ):
        keys = set(dtypes.keys())

        if keys != set(fixed_shapes.keys()):
            raise ValueError("Non-matching keys in dtypes and fixed_shapes")

        dtypes = {
            k: str(v) if isinstance(v, torch.dtype) else v for k, v in dtypes.items()
        }

        if isinstance(path, str):
            path = Path(path)

        if path.exists():
            raise ValueError(f"folder already exists at {path}")
        path.mkdir(parents=True)

        metadata = DictPilerMetadata(
            keys=keys,
        )

        pilers = {
            k: Piler.create(
                path / k,
                dtypes[k],
                fixed_shapes[k],
                num_piles,
                compress,
            )
            for k in keys
        }
        dict_piler = cls(
            # metadata=metadata,
            # piler_metadata=first_piler.metadata,
            # pilers=pilers,
            path=path,
            use_async_distribute=use_async_distribute,
            readonly=False,
        )
        assert not cls.get_metadata_path(path).exists()
        cls.get_metadata_path(path).write_text(metadata.model_dump_json())
        dict_piler.pilers = pilers

        return dict_piler

    @classmethod
    def open(
        cls,
        path: Path,
        use_async_distribute: bool = True,
    ):
        dict_piler = DictPiler(
            path=path,
            use_async_distribute=use_async_distribute,
            readonly=True,
        )

        return dict_piler

    @cached_property
    def metadata(self):
        metadata_path = self.get_metadata_path(self.path)
        if not metadata_path.exists():
            raise ValueError(f"DictPiler metadata not found at {metadata_path}")
        return DictPilerMetadata.model_validate_json(metadata_path.read_text())

    @cached_property
    def piler_metadata(self):
        return next(iter(self.pilers.values())).metadata

    @cached_property
    def pilers(self):
        pilers = {k: Piler.open(self.path / k) for k in self.metadata.keys}

        first_piler = next(iter(pilers.values()))

        for piler in pilers.values():
            if first_piler.metadata.num_piles != piler.metadata.num_piles:
                raise ValueError(
                    f"Piler {piler.path} does not match first piler {first_piler.path}: {piler.metadata.num_piles} != {first_piler.metadata.num_piles}"
                )
            if first_piler.shape[0] != piler.shape[0]:
                raise ValueError(
                    f"Piler {piler.path} shape does not match first piler {first_piler.path}: {piler.shape[0]} != {first_piler.shape[0]}"
                )
            if first_piler.metadata.compression != piler.metadata.compression:
                raise ValueError(
                    f"Piler {piler.path} compression does not match first piler {first_piler.path}: {piler.metadata.compression} != {first_piler.metadata.compression}"
                )
        return pilers

    @classmethod
    def get_metadata_path(cls, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        return path / "dictpiler_metadata.json"

    def distribute(
        self, tensors: dict[str, torch.Tensor], indexer: torch.Tensor | None = None
    ):
        if indexer is None:
            i = torch.randint(
                0,
                self.piler_metadata.num_piles,
                [tensors[next(iter(self.metadata.keys))].shape[0]],
            )
        else:
            i = indexer
        assert all(t.shape[0] == i.shape[0] for t in tensors.values())
        assert self.metadata.keys == set(tensors.keys())
        if self.use_async_distribute:
            for k, t in tensors.items():
                self.pilers[k].distribute_async(t, i)
        else:
            for k, t in tensors.items():
                self.pilers[k].distribute(t, i)
        return i

    def shuffle_piles(self):
        if self.readonly:
            raise ValueError("Cannot write to a readonly Piler")
        i = iter(self.pilers.values())
        piler = next(i)
        catlens = {k: piler.piles.get(k).cat_len for k in piler.piles.keys()}

        perms = {k: torch.randperm(catlen) for k, catlen in catlens.items()}
        for k, p in self.pilers.items():
            p.shuffle_piles(perms)

    def __getitem__(self, i) -> DictBatch:
        data = {k: self.pilers[k][i] for k in self.metadata.keys}
        assert isinstance(data, dict)
        assert all(isinstance(v, torch.Tensor) for v in data.values())
        return DictBatch(data=cast(dict[str, torch.Tensor], data))

    @overload
    def batch_generator(
        self,
        batch_size,
        yield_dicts: Literal[False] = False,
        id=None,
        nw=None,
        num_epochs: int | None = 1,
        shuffle: bool = True,
        shuffle_piles_order: bool = False,
    ) -> Generator[DictBatch]: ...

    @overload
    def batch_generator(
        self,
        batch_size,
        yield_dicts: Literal[True],
        id=None,
        nw=None,
        num_epochs: int | None = 1,
        shuffle: bool = True,
        shuffle_piles_order: bool = False,
    ) -> Generator[dict[str, torch.Tensor]]: ...
    @torch.inference_mode()
    def batch_generator(
        self,
        batch_size,
        yield_dicts: bool = False,
        id=None,
        nw=None,
        num_epochs: int | None = 1,
        shuffle: bool = True,
        shuffle_piles_order: bool = True,
    ):
        if not (id == nw == None or id is not None and nw is not None):
            raise ValueError("id and nw must be either both None or both not None")
        id = id or 0
        nw = nw or 1
        if num_epochs is not None:
            epoch_gen = range(num_epochs)
        else:
            epoch_gen = itertools.count()

        spares: list[DictBatch] = []
        nspare = 0
        perm = None

        def newperm(pile):
            nonlocal perm
            if shuffle:
                perm = torch.randperm(len(pile))

        def getslice(pile, a, b):
            nonlocal perm
            if perm is None:
                return pile[a:b]
            return pile[perm[a:b]]

        for epoch in epoch_gen:
            if epoch != 0:
                print(f"finished epoch {epoch - 1}")
            for p in shuffled_range(
                (id) % nw,
                self.piler_metadata.num_piles,
                nw,
                shuffle=shuffle and shuffle_piles_order,
            ):
                pile = self[p]
                newperm(pile)
                # perm = torch.randperm(len(pile))
                # below we clone before yielding to prevent yielding a view of the pile.
                # if a yielded view were to get pinned by the consumer of this,
                # (eg a dataloader), the entire mmapped pile would get pinned as well
                for i in shuffled_range(
                    0, len(pile) // batch_size * batch_size, batch_size, shuffle=shuffle
                ):
                    pile_slice = getslice(pile, i, i + batch_size).clone()
                    yield (pile_slice.data if yield_dicts else pile_slice)
                spare = getslice(
                    pile, len(pile) // batch_size * batch_size, None
                ).clone()
                if len(spare) > 0:
                    spares.append(spare)
                    nspare += len(spare)
                    if nspare >= batch_size:
                        consolidated = DictBatch.cat_list(spares, dim=0)
                        for i in shuffled_range(
                            0, len(consolidated) // batch_size * batch_size, batch_size
                        ):
                            consolidated_slice = consolidated[
                                i : i + batch_size
                            ].clone()  # no getslice because this was already permed
                            yield (
                                consolidated_slice.data
                                if yield_dicts
                                else consolidated_slice
                            )
                        spare = consolidated[
                            len(consolidated) // batch_size * batch_size :
                        ]
                        spares = [spare]
                        nspare = len(spare)
                for piler in self.pilers.values():
                    print("checking through pilers")

                    if not piler.piles:
                        print(f"not deleting item from piler cache: {p}")
                        continue
                        try:
                            del piler.piles.cache[str(p)]
                        except KeyError as e:
                            print(
                                f"Warning, deleting item from piler cache failed: {e}"
                            )

    @overload
    def sized_generator(
        self,
        yield_dicts: Literal[False] = False,
        id=None,
        nw=None,
    ) -> tuple[Callable[[int], tuple[DictBatch, int]], int]: ...

    @overload
    def sized_generator(
        self,
        yield_dicts: Literal[True],
        id=None,
        nw=None,
    ) -> tuple[Callable[[int], tuple[dict[str, torch.Tensor] | None, int]], int]: ...

    def sized_generator(
        self,
        yield_dicts: bool = False,
        id=None,
        nw=None,
        return_last_batch: bool = False,
    ):
        if not (id == nw == None or id is not None and nw is not None):
            raise ValueError("id and nw must be either both None or both not None")
        id = id or 0
        nw = nw or 1

        def piles_gen():
            for p in range(id % nw, self.piler_metadata.num_piles, nw):
                yield self[p]
            # below we clone before yielding to prevent yielding a view of the pile.
            # if a yielded view were to get pinned by the consumer of this,
            # (eg a dataloader), the entire mmapped pile would get pinned as well

        piler0 = next(iter(self.pilers.values()))
        num_samples_total = sum(
            piler0.shapes[p][0]
            for p in range(id % nw, self.piler_metadata.num_piles, nw)
        )

        num_returned = 0

        i = 0
        piles = piles_gen()
        pile = next(piles)
        done = False

        def get(batch_size):
            nonlocal i, pile, piles, done, num_samples_total, num_returned
            if done:
                raise StopIteration("No more batches")
            res = None
            while len(pile) - i < batch_size:
                try:
                    pile = DictBatch.cat_list([pile[i:].clone(), next(piles)])
                    i = 0
                except StopIteration as e:
                    if not return_last_batch:
                        raise e
                    done = True
                    num_returned += len(pile) - i
                    assert num_returned == num_samples_total
                    return pile[i:].clone(), 0
            res = (
                pile[i : i + batch_size].clone().data
                if yield_dicts
                else pile[i : i + batch_size].clone()
            )
            i += batch_size
            num_returned += batch_size
            return res, num_samples_total - num_returned

        return get, num_samples_total

    @property
    def num_piles(self):
        n = next(iter(self.pilers.values())).num_piles
        assert all(piler.num_piles == n for piler in self.pilers.values())
        return n

    @property
    def num_samples(self):
        samples = next(iter(self.pilers.values())).shape[0]
        assert all(piler.shape[0] == samples for piler in self.pilers.values())
        return samples

    @property
    def shapes(self) -> dict[str, list[list[int]]]:
        return {k: piler.shapes for k, piler in self.pilers.items()}

    @cached_property
    def pile_indices(self) -> list[int]:
        assert self.readonly

        l = [[shape[0] for shape in piler.shapes] for piler in self.pilers.values()]
        assert all(e == l[0] for e in l)
        return l[0]

    def _index_by_sample(self, index: int | slice) -> DictBatch:
        if isinstance(index, int):
            pile_idx, sample_idx = convert_sample_index_to_pile_pair(
                index, self.pile_indices
            )
            return self[pile_idx][sample_idx]
        start, stop, step = index.indices(self.num_samples)
        start_pile, start_sample = convert_sample_index_to_pile_pair(
            start, self.pile_indices
        )
        stop_pile, _ = convert_sample_index_to_pile_pair(stop, self.pile_indices)
        dist = stop - start
        data = self[start_pile : stop_pile + 1]
        assert data.batch_size >= dist + start_sample, (
            data.batch_size,
            dist,
            start_sample,
        )
        return data[start_sample : start_sample + dist]

    @property
    def sample_indexer(self) -> "DictPilerSampleIndexer":
        return DictPilerSampleIndexer(piler=self)


def convert_sample_index_to_pile_pair(
    idx: int, pile_sizes: list[int]
) -> tuple[int, int]:
    pile_idx = idx
    for i, pile_size in enumerate(pile_sizes):
        if pile_idx < pile_size:
            return i, pile_idx
        pile_idx -= pile_size
    raise ValueError(f"Index {idx} is out of bounds for pile sizes {pile_sizes}")


@define
class DictPilerSampleIndexer:
    piler: DictPiler

    def __getitem__(self, index: int | slice) -> DictBatch:
        return self.piler._index_by_sample(index)


def main():
    test_path = Path("testdata")
    dp = DictPiler.create(
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
