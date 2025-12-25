from functools import cached_property
from pathlib import Path

import torch
from attrs import define
from jaxtyping import Float
from paramsight import get_resolved_typevars_for_base, takes_alias
from safetensors.torch import load_file, save_file

from saeco.data import DictBatch
from saeco.data.dict_batch import DictBatch
from saeco.evaluation.storage.saved_acts_config import CachingConfig

from ...data.storage.sparse_safetensors import load_sparse_tensor, save_sparse_tensor
from ..filtered import Filter, FilteredTensor
from ..named_filter import NamedFilter


@define
class Chunk[InputsT: torch.Tensor | DictBatch]:
    idx: int
    path: Path
    loaded_input_data: InputsT | None = None
    dense_acts: Float[torch.Tensor, "doc seq d_dict"] | None = None
    sparse_acts: Float[torch.Tensor, "nnz 3"] | None = None
    sparsify_batch_size: int = 100
    named_filter: NamedFilter | None = None

    # default_sparse: bool = True
    @cached_property
    def cfg(self) -> CachingConfig[InputsT]:
        cfg_path = self.path / CachingConfig.STANDARD_FILE_NAME
        if not cfg_path.exists():
            raise FileNotFoundError(f"Could not find caching config at {cfg_path}")
        return CachingConfig.model_validate_json((cfg_path).read_text())

    def sparsify(self):
        if self.sparse_acts is not None:
            return
        assert self.dense_acts is not None
        self.sparse_acts = self.dense_acts.to_sparse_coo()

    @property
    def dense_disk_storage_path(self):
        return self.dense_path.parent / f"{self.dense_path.stem}.bin"

    def densify(self):
        assert self.sparse_acts is not None
        if self.dense_acts is not None:
            return self
        self.dense_acts = self.sparse_acts.to_dense()

    @property
    def sparse_path(self):
        return self.path / f"sparse_acts_{self.idx}.safetensors"

    @property
    def dense_path(self):
        return self.path / f"acts_{self.idx}.safetensors"

    @property
    def tokens_path(self):
        return self.path / f"tokens_{self.idx}.safetensors"

    def save_sparse(self):
        if self.sparse_acts is None:
            self.sparsify()
        assert not self.sparse_path.exists()
        save_sparse_tensor(self.sparse_acts, self.sparse_path)

    def save_dense(self):
        assert self.dense_acts is not None
        assert not self.dense_path.exists()
        save_file({"acts": self.dense_acts}, self.dense_path)

    def save_tokens(self):
        loaded_input_data = self.loaded_input_data
        assert loaded_input_data is not None
        assert not self.tokens_path.exists()
        if isinstance(loaded_input_data, DictBatch):
            loaded_input_data.save_as_safetensors(self.tokens_path)
        else:
            assert isinstance(loaded_input_data, torch.Tensor)
            save_file({"tokens": loaded_input_data.contiguous()}, self.tokens_path)

    def load_sparse(self):
        self.sparse_acts = self.read_sparse_raw()

    def load_dense(self):
        self.dense_acts = self.read_dense_raw()

    def load_tokens(self):
        self.loaded_input_data = self.read_tokens_raw()

    def _to_filtered(self, chunk_tensor: torch.Tensor | DictBatch):
        sl = slice(
            self.cfg.docs_per_chunk * self.idx, self.cfg.docs_per_chunk * (self.idx + 1)
        )
        mask = self.named_filter.filter[sl] if self.named_filter is not None else None

        # if isinstance(chunk_tensor, DictBatch):
        #     assert all(
        #         v.shape[0] == self.cfg.docs_per_chunk for v in chunk_tensor.values()
        #     )
        #     ft_dict = {
        #         k: FilteredTensor.from_unmasked_value(
        #             v,
        #             filter_obj=Filter(
        #                 [sl],
        #                 mask=mask,
        #                 shape=[self.cfg.num_docs, *v.shape[1:]],
        #             ),
        #             presliced=True,
        #         )
        #         for k, v in chunk_tensor.items()
        #     }
        #     return chunk_tensor.__class__.construct_with_other_data(
        #         ft_dict, chunk_tensor._get_other_dict()
        #     )

        assert chunk_tensor.shape[0] == self.cfg.docs_per_chunk
        filt = Filter(
            [sl],
            mask=mask,
            shape=(self.cfg.num_docs,),
        )
        return FilteredTensor.from_unmasked_value(
            chunk_tensor, filter_obj=filt, presliced=True
        )

    def read_sparse_raw(self):
        if self.sparse_acts is not None:
            return self.sparse_acts
        return load_sparse_tensor(self.sparse_path)

    def read_dense_raw(self):
        if self.dense_acts is not None:
            return self.dense_acts
        return load_file(self.dense_path)["acts"]

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
        if issubclass(self.input_data_cls, DictBatch):
            return self.input_data_cls.load_from_safetensors(self.tokens_path)
        loaded = load_file(self.tokens_path)
        assert set(loaded.keys()) == {"tokens"}
        assert self.input_data_cls == torch.Tensor
        tokens = loaded["tokens"]
        assert isinstance(tokens, self.input_data_cls)
        return tokens

    def read_sparse(self):
        return self._to_filtered(self.read_sparse_raw())

    def read_dense(self):
        return self._to_filtered(self.read_dense_raw())

    def read_tokens(self):
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
    ) -> "Chunk":
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
            try:
                chunk = cls.load_from_dir(
                    path, i, load_sparse_only, lazy=lazy, filter_obj=filter_obj
                )
                if not chunk.exists:
                    break
                yield chunk
                i += 1
            except Exception:
                break

    @takes_alias
    @classmethod
    def load_chunks_from_dir(
        cls,
        path: Path,
        load_sparse_only: bool = False,
        lazy: bool = False,
        filter_obj: NamedFilter | None = None,
    ):
        return list(
            cls.chunks_from_dir_iter(
                path, load_sparse_only, lazy=lazy, filter_obj=filter_obj
            )
        )

    def load(self, load_sparse_only: bool = False):
        assert self.tokens_path.exists()
        assert (
            self.dense_path.exists() and not load_sparse_only
        ) or self.sparse_path.exists()
        if self.dense_path.exists() and not load_sparse_only:
            self.load_dense()
        if self.sparse_path.exists():
            self.load_sparse()
        self.load_tokens()

    @property
    def exists(self):
        return self.tokens_path.exists() and (
            self.dense_path.exists() or self.sparse_path.exists()
        )

    @property
    def acts(self) -> FilteredTensor:
        return self.read_sparse()

    @property
    def tokens_batch(self):
        return self.read_tokens()

    @property
    def tokens(self) -> FilteredTensor:
        return self.tokens_batch

    @property
    def acts_raw(self):
        if self.named_filter is not None:
            raise ValueError("Accessing raw values, but the filter is not None")
        return self.read_sparse_raw()

    @property
    def tokens_raw(self):
        if self.named_filter is not None:
            raise ValueError("Accessing raw values, but the filter is not None")
        return self.read_tokens_raw()

    @property
    def acts_raw_unsafe(self):
        return self.read_sparse_raw()

    @property
    def tokens_raw_unsafe(self):
        return self.read_tokens_raw()
