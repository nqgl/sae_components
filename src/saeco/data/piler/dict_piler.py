from enum import Enum
from pathlib import Path

from typing import (
    Any,
    ClassVar,
    Generator,
    Iterable,
    Literal,
    cast,
    List,
    overload,
    Sequence,
    Union,
    Mapping,
)
import einops

import torch
import tqdm
from attrs import define, field
from pydantic import BaseModel

from saeco.data.piler import Piler, PilerMetadata
from saeco.data.storage.compressed_safetensors import CompressionType

from saeco.data.storage.growing_disk_tensor_collection import (
    GrowingDiskTensorCollection,
)

from typing_extensions import Self


@define
class DictBatch:
    """
        Represents a batch of data from multiple locations in a model, all aligned to the
    same activation tokens/inference steps.

        Indexing with a string returns the tensor associated with that key (location).

        Indexing with an integer, slice, list of indices, or a torch.Tensor applies the
        indexing operation to each tensor in the batch, returning a new DictBatch.
    """

    data: dict[str, torch.Tensor]

    @overload
    def __getitem__(self, i: str) -> torch.Tensor: ...
    @overload
    def __getitem__(
        self, i: int | slice | list[int] | torch.Tensor | tuple[slice, ...]
    ) -> Self: ...
    def __getitem__(self, i):
        if isinstance(i, str):
            return self.data[i]
        return self.__class__({k: v[i] for k, v in self.data.items()})

    def to(self, *targets):
        return self.__class__({k: v.to(*targets) for k, v in self.data.items()})

    def cuda(self):
        return self.to("cuda")

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def cat(self, other: Self):
        assert self.keys() == other.keys()
        return self.__class__(
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
    def cat_list(
        cls, batches: list[Self], dim: int = 0, overwite_kwargs: dict | None = None
    ):
        overwite_kwargs = overwite_kwargs or {}
        other_data_0: dict[str, Any] = batches[0].get_other_data()
        keys0 = set(other_data_0.keys())
        overwritten_keys = set(overwite_kwargs.keys())
        normal_keys = keys0 - overwritten_keys

        for batch in batches:
            assert normal_keys == set(batch.get_other_data().keys() - overwritten_keys)
            for normal_key in normal_keys:
                # assert equal values
                ...
        other_data = {
            **other_data_0,
            **overwite_kwargs,
        }  # this ordering makes 2nd overwrite

        keys = batches[0].data.keys()
        assert all(b.data.keys() == keys for b in batches)
        return cls({k: torch.cat([b.data[k] for b in batches], dim=dim) for k in keys})

    def einops_rearrange(self, pattern: str, **kwargs):
        return self.__class__(
            {k: einops.rearrange(v, pattern, **kwargs) for k, v in self.data.items()}
        )

    @classmethod
    def construct_with_other_data(
        cls, data: dict[str, torch.Tensor], other_data: dict[str, Any] | None = None
    ):
        return cls(data, **(other_data or {}))

    def copy_with_data(self, data: dict[str, torch.Tensor]):
        other_data = self.get_other_data()
        return self.__class__(data, **other_data)

    OTHER_DATA_FIELDS: ClassVar[tuple[str, ...]] = ()

    def get_other_data(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.OTHER_DATA_FIELDS}


class DictPilerMetadata(BaseModel):
    keys: set[str]


@define
class DictPiler:
    metadata: DictPilerMetadata
    piler_metadata: PilerMetadata
    pilers: dict[str, Piler]
    path: Path
    use_async_distribute: bool = (
        True  # this is a flag just bc it's a new/experimental feature
    )
    readonly: bool = False

    @classmethod
    def create(
        cls,
        path: Union[str, Path],
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

        first_piler = next(iter(pilers.values()))

        dict_piler = DictPiler(
            metadata,
            first_piler.metadata,
            pilers,
            path,
            use_async_distribute,
            False,
        )

        cls.get_metadata_path(path).write_text(metadata.model_dump_json())

        return dict_piler

    @classmethod
    def open(
        cls,
        path: Union[str, Path],
        use_async_distribute: bool = True,
        # we could allow options to be passed in here and then assert that they match the properties of the opened piler
        # not sure that's necessary though
    ):
        metadata_path = cls.get_metadata_path(path)

        if not metadata_path.exists():
            raise ValueError(f"DictPiler metadata not found at {metadata_path}")

        metadata = DictPilerMetadata.model_validate_json(metadata_path.read_text())

        pilers = {k: Piler.open(path / k) for k in metadata.keys}

        first_piler = next(iter(pilers.values()))

        assert all(
            first_piler.metadata.num_piles == piler.metadata.num_piles
            and first_piler.metadata.dtype == piler.metadata.dtype
            and first_piler.metadata.compression == piler.metadata.compression
            and first_piler.metadata.fixed_shape == piler.metadata.fixed_shape
            for piler in pilers.values()
        )

        dict_piler = DictPiler(
            metadata,
            first_piler.metadata,
            pilers,
            path,
            use_async_distribute,
            readonly=True,
        )

        return dict_piler

    @classmethod
    def get_metadata_path(cls, path: Union[str, Path]):
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
    ) -> Generator[DictBatch, None, None]: ...

    @overload
    def batch_generator(
        self,
        batch_size,
        yield_dicts: Literal[True],
        id=None,
        nw=None,
    ) -> Generator[dict[str, torch.Tensor], None, None]: ...

    def batch_generator(
        self,
        batch_size,
        yield_dicts: bool = False,
        id=None,
        nw=None,
    ):
        if not (id == nw == None or id is not None and nw is not None):
            raise ValueError("id and nw must be either both None or both not None")
        id = id or 0
        nw = nw or 1

        spares: list[DictBatch] = []
        nspare = 0
        for p in range(id % nw, self.piler_metadata.num_piles, nw):
            pile = self[p]
            for i in range(0, len(pile) // batch_size * batch_size, batch_size):
                yield (
                    pile[i : i + batch_size].data
                    if yield_dicts
                    else pile[i : i + batch_size]
                )
            spare = pile[len(pile) // batch_size * batch_size :]
            if len(spare) > 0:
                spares.append(spare)
                nspare += len(spare)
                if nspare > batch_size:
                    consolidated = DictBatch.cat_list(spares, dim=0)
                    for i in range(
                        0, len(consolidated) // batch_size * batch_size, batch_size
                    ):
                        yield (
                            consolidated[i : i + batch_size].data
                            if yield_dicts
                            else consolidated[i : i + batch_size]
                        )
                    spare = consolidated[len(consolidated) // batch_size * batch_size :]
                    spares = [spare]
                    nspare = len(spare)

    def as_dataset(self, batch_size, converter=None):
        return PilerDataset(self, batch_size, converter)

    @property
    def num_piles(self):
        return next(iter(self.pilers.values())).num_piles


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
            gen = self.piler.batch_generator(
                self.batch_size, yield_dicts=self.converter is None
            )

        else:
            id = worker_info.id
            nw = worker_info.num_workers
            assert id % nw == id, (id, nw)
            print("worker", id, nw, worker_info)
            gen = self.piler.batch_generator(
                batch_size=self.batch_size,
                yield_dicts=self.converter is None,
                id=id,
                nw=nw,
            )
        if self.converter is not None:
            return self.converter(gen)
        return gen


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
