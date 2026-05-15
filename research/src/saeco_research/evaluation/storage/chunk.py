from __future__ import annotations

from functools import cached_property
from pathlib import Path

import torch
from attrs import define
from jaxtyping import Float
from paramsight import get_resolved_typevars_for_base, takes_alias
from safetensors.torch import load_file, save_file

from saeco.data.dict_batch import DictBatch
from saeco.misc.utils import chill_issubclass

from saeco.data.storage.sparse_safetensors import load_sparse_tensor, save_sparse_tensor
from ..filtered import Filter, FilteredTensor
from ..named_filter import NamedFilter
from .cache_config import CacheConfig


@define
class Chunk[InputsT: torch.Tensor | DictBatch]:
    idx: int
    path: Path
    loaded_input_data: InputsT | None = None
    dense_acts: Float[torch.Tensor, "doc seq d_dict"] | None = None
    sparse_acts: Float[torch.Tensor, "nnz 3"] | None = None
    sparsify_batch_size: int = 100
    named_filter: NamedFilter | None = None

    @cached_property
    def cfg(self) -> CacheConfig[InputsT]:
        cfg_path = self.path / CacheConfig.STANDARD_FILE_NAME
        if not cfg_path.exists():
            raise FileNotFoundError(f"Could not find cache config at {cfg_path}")
        return CacheConfig.model_validate_json(cfg_path.read_text())

    def sparsify(self) -> None:
        if self.sparse_acts is not None:
            return
        if self.dense_acts is None:
            raise ValueError("Cannot sparsify without dense_acts loaded")
        self.sparse_acts = self.dense_acts.to_sparse_coo()

    def densify(self) -> None:
        if self.dense_acts is not None:
            return
        if self.sparse_acts is None:
            raise ValueError("Cannot densify without sparse_acts loaded")
        self.dense_acts = self.sparse_acts.to_dense()

    @property
    def sparse_path(self) -> Path:
        return self.path / f"sparse_acts_{self.idx}.safetensors"

    @property
    def dense_path(self) -> Path:
        return self.path / f"acts_{self.idx}.safetensors"

    @property
    def tokens_path(self) -> Path:
        return self.path / f"tokens_{self.idx}.safetensors"

    def save_sparse(self) -> None:
        if self.sparse_acts is None:
            self.sparsify()
        if self.sparse_path.exists():
            raise FileExistsError(self.sparse_path)
        save_sparse_tensor(self.sparse_acts, self.sparse_path)  # type: ignore[arg-type]

    def save_dense(self) -> None:
        if self.dense_acts is None:
            raise ValueError("dense_acts is None")
        if self.dense_path.exists():
            raise FileExistsError(self.dense_path)
        save_file({"acts": self.dense_acts}, self.dense_path)

    def save_tokens(self) -> None:
        if self.loaded_input_data is None:
            raise ValueError("loaded_input_data is None")
        if self.tokens_path.exists():
            raise FileExistsError(self.tokens_path)
        if isinstance(self.loaded_input_data, DictBatch):
            self.loaded_input_data.save_as_safetensors(self.tokens_path)
        else:
            save_file({"tokens": self.loaded_input_data.contiguous()}, self.tokens_path)

    def _to_filtered(self, chunk_tensor: torch.Tensor | DictBatch) -> FilteredTensor:
        sl = slice(
            self.cfg.docs_per_chunk * self.idx,
            self.cfg.docs_per_chunk * (self.idx + 1),
        )
        mask = self.named_filter.filter[sl] if self.named_filter is not None else None
        filt = Filter(slices=(sl,), mask=mask, shape=(self.cfg.num_docs,))
        return FilteredTensor.from_unmasked_value(
            chunk_tensor, filter_obj=filt, presliced=True
        )

    def read_sparse_raw(self) -> torch.Tensor:
        return (
            self.sparse_acts
            if self.sparse_acts is not None
            else load_sparse_tensor(self.sparse_path)
        )

    def read_dense_raw(self) -> torch.Tensor:
        return (
            self.dense_acts
            if self.dense_acts is not None
            else load_file(self.dense_path)["acts"]
        )

    @takes_alias
    @classmethod
    def get_input_data_cls(cls) -> type[InputsT]:
        return get_resolved_typevars_for_base(cls, Chunk)[0]  # type: ignore

    @property
    def input_data_cls(self) -> type[InputsT]:
        return self.get_input_data_cls()

    def read_tokens_raw(self) -> InputsT:
        if self.loaded_input_data is not None:
            return self.loaded_input_data
        if chill_issubclass(self.input_data_cls, DictBatch):
            return self.input_data_cls.load_from_safetensors(self.tokens_path)
        loaded = load_file(self.tokens_path)
        if set(loaded.keys()) != {"tokens"}:
            raise ValueError(f"Unexpected tokens keys: {loaded.keys()}")
        return loaded["tokens"]  # type: ignore[return-value]

    def read_sparse(self) -> FilteredTensor:
        return self._to_filtered(self.read_sparse_raw())

    def read_dense(self) -> FilteredTensor:
        return self._to_filtered(self.read_dense_raw())

    def read_tokens(self) -> FilteredTensor:
        return self._to_filtered(self.read_tokens_raw())

    @takes_alias
    @classmethod
    def load_from_dir(
        cls,
        path: Path,
        index: int,
        load_sparse_only: bool = False,
        lazy: bool = False,
        filter_obj: NamedFilter | None = None,
    ) -> Chunk:
        inst = cls(path=path, idx=index, named_filter=filter_obj)
        if lazy:
            return inst
        inst.load(load_sparse_only=load_sparse_only)
        return inst

    @takes_alias
    @classmethod
    def chunks_from_dir_iter(
        cls,
        path: Path,
        load_sparse_only: bool = False,
        lazy: bool = False,
        filter_obj: NamedFilter | None = None,
    ):
        i = 0
        while True:
            inst = cls.load_from_dir(
                path, i, load_sparse_only, lazy=lazy, filter_obj=filter_obj
            )
            if not inst.exists:
                break
            yield inst
            i += 1

    @takes_alias
    @classmethod
    def load_chunks_from_dir(
        cls,
        path: Path,
        load_sparse_only: bool = False,
        lazy: bool = False,
        filter_obj: NamedFilter | None = None,
    ) -> list[Chunk]:
        return list(
            cls.chunks_from_dir_iter(
                path, load_sparse_only, lazy=lazy, filter_obj=filter_obj
            )
        )

    def load(self, load_sparse_only: bool = False) -> None:
        if not self.tokens_path.exists():
            raise FileNotFoundError(self.tokens_path)
        if self.dense_path.exists() and not load_sparse_only:
            self.dense_acts = self.read_dense_raw()
        if self.sparse_path.exists():
            self.sparse_acts = self.read_sparse_raw()
        if not (self.dense_path.exists() or self.sparse_path.exists()):
            raise FileNotFoundError("Chunk has neither dense nor sparse acts")
        self.loaded_input_data = self.read_tokens_raw()

    @property
    def exists(self) -> bool:
        return self.tokens_path.exists() and (
            self.dense_path.exists() or self.sparse_path.exists()
        )

    @property
    def acts(self) -> FilteredTensor:
        return self.read_sparse()

    @property
    def tokens(self) -> FilteredTensor:
        return self.read_tokens()

    @property
    def doc_ids(self) -> torch.Tensor:
        start = self.cfg.docs_per_chunk * self.idx
        stop = self.cfg.docs_per_chunk * (self.idx + 1)
        doc_ids = torch.arange(start, stop, dtype=torch.long)
        if self.named_filter is not None and self.named_filter.filter is not None:
            mask = self.named_filter.filter[start:stop].to(doc_ids.device)
            doc_ids = doc_ids[mask]
        return doc_ids
