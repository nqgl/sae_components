from functools import cached_property

from pathlib import Path

import torch
from attrs import define, field
from jaxtyping import Float, Int
from saeco.evaluation.saved_acts_config import CachingConfig
from safetensors.torch import load_file, save_file

from ..filtered import Filter, FilteredTensor
from ..filtered_evaluation import NamedFilter
from .sparse_safetensors import load_sparse_tensor, save_sparse_tensor


@define
class Chunk:
    idx: int
    path: Path
    loaded_tokens: Int[torch.Tensor, "doc seq"] | None = None
    _dense_acts: Float[torch.Tensor, "doc seq d_dict"] | None = None
    _sparse_acts: Float[torch.Tensor, "nnz 3"] | None = None
    _cfg: CachingConfig | None = None
    sparsify_batch_size: int = 100
    _filter: NamedFilter | None = None

    # default_sparse: bool = True
    @property
    def cfg(self):
        if self._cfg is not None:
            return self._cfg
        cfg_path = self.path / CachingConfig.STANDARD_FILE_NAME
        if cfg_path.exists():
            self._cfg = CachingConfig.model_validate_json((cfg_path).read_text())
        return self._cfg

    def sparsify(self):
        if self._sparse_acts is not None:
            return
        assert self._dense_acts is not None
        self._sparse_acts = self._dense_acts.to_sparse_coo()
        # if self.dense_acts.shape[0] > self.sparsify_batch_size:
        #     indices = []
        #     values = []
        #     for i in range(0, self.dense_acts.shape[0], self.sparsify_batch_size):
        #         batch = self.dense_acts[i : i + self.sparsify_batch_size]
        #         sparse_batch = batch.to_sparse_coo()
        #         indices.append(sparse_batch.indices())
        #         values.append(sparse_batch.values())
        #     indices = torch.cat(indices, dim=1)
        #     values = torch.cat(values)
        #     self.sparse_acts = torch.sparse_coo_tensor(
        #         indices, values, self.dense_acts.shape
        #     )
        # else:

    # def make_dense_disk_storage(self, seq_len, d_dict, dtype_bytes=4):
    #     shape = [self.cfg.docs_per_chunk, seq_len, d_dict]
    #     numel = shape[0] * shape[1] * shape[2]
    #     assert not self.dense_disk_storage_path.exists()
    #     storage = torch.FloatTensor(
    #         torch.UntypedStorage.from_file(
    #             str(self.dense_disk_storage_path),
    #             shared=True,
    #             nbytes=numel * dtype_bytes,
    #         )
    #     )
    #     storage.reshape(shape)

    @property
    def dense_disk_storage_path(self):
        return self.dense_path.parent / f"{self.dense_path.stem}.bin"

    def densify(self):
        assert self._sparse_acts is not None
        if self._dense_acts is not None:
            return self
        self._dense_acts = self._sparse_acts.to_dense()

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
        if self._sparse_acts is None:
            self.sparsify()
        assert not self.sparse_path.exists()
        save_sparse_tensor(self._sparse_acts, self.sparse_path)

    def save_dense(self):
        assert self._dense_acts is not None
        assert not self.dense_path.exists()
        save_file({"acts": self._dense_acts}, self.dense_path)

    def save_tokens(self):
        assert self.loaded_tokens is not None
        assert not self.tokens_path.exists()
        save_file({"tokens": self.loaded_tokens}, self.tokens_path)

    def load_sparse(self):
        self._sparse_acts = self.read_sparse_raw()

    def load_dense(self):
        self._dense_acts = self.read_dense_raw()

    def load_tokens(self):
        self.loaded_tokens = self.read_tokens_raw()

    def _to_filtered(self, chunk_tensor: torch.Tensor):
        assert chunk_tensor.shape[0] == self.cfg.docs_per_chunk
        sl = slice(
            self.cfg.docs_per_chunk * self.idx, self.cfg.docs_per_chunk * (self.idx + 1)
        )

        filt = Filter(
            [sl],
            mask=sl.apply(self._filter.filter) if self._filter is not None else None,
            shape=[self.cfg.num_docs, *chunk_tensor.shape[1:]],
        )
        return FilteredTensor.from_unmasked_value(
            chunk_tensor, filter=filt, presliced=True
        )

    def read_sparse_raw(self):
        if self._sparse_acts is not None:
            return self._sparse_acts
        return load_sparse_tensor(self.sparse_path)

    def read_dense_raw(self):
        if self._dense_acts is not None:
            return self._dense_acts
        return load_file(self.dense_path)["acts"]

    def read_tokens_raw(self):
        if self.loaded_tokens is not None:
            return self.loaded_tokens
        return load_file(self.tokens_path)["tokens"]

    def read_sparse(self):
        return self._to_filtered(self.read_sparse_raw())

    def read_dense(self):
        return self._to_filtered(self.read_dense_raw())

    def read_tokens(self):
        return self._to_filtered(self.read_tokens_raw())

    @classmethod
    def load_from_dir(
        cls, path, index, load_sparse_only: bool = False, lazy=False, filter=None
    ) -> "Chunk":
        inst = cls(path=path, idx=index, filter=filter)
        if lazy:
            return inst
        inst.load(load_sparse_only=load_sparse_only)
        return inst

    @classmethod
    def chunks_from_dir_iter(
        cls, path, load_sparse_only: bool = False, lazy=False, filter=None
    ):
        i = 0
        while True:
            try:
                chunk = cls.load_from_dir(
                    path, i, load_sparse_only, lazy=lazy, filter=filter
                )
                if not chunk.exists:
                    break
                yield chunk
                i += 1
            except:
                break

    @classmethod
    def load_chunks_from_dir(
        cls, path, load_sparse_only: bool = False, lazy=False, filter=None
    ):
        return list(
            cls.chunks_from_dir_iter(path, load_sparse_only, lazy=lazy, filter=filter)
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
    def acts(self):
        return self.read_sparse()

    @property
    def tokens(self):
        return self.read_tokens()

    @property
    def acts_raw(self):
        if self._filter is not None:
            raise ValueError("Accessing raw values, but the filter is not None")
        return self.read_sparse_raw()

    @property
    def tokens_raw(self):
        if self._filter is not None:
            raise ValueError("Accessing raw values, but the filter is not None")
        return self.read_tokens_raw()

    @property
    def acts_raw_unsafe(self):
        return self.read_sparse_raw()

    @property
    def tokens_raw_unsafe(self):
        return self.read_tokens_raw()

        # if not self.default_sparse:
        #     if self.dense_acts is None:
        #         return self.read_dense()
        #     return self.dense_acts
