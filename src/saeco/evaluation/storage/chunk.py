from .sparse_safetensors import load_sparse_tensor, save_sparse_tensor
from saeco.evaluation.saved_acts_config import CachingConfig

import torch
from attrs import define, field
from jaxtyping import Float, Int
from safetensors.torch import load_file, save_file
from functools import cached_property

from pathlib import Path


@define
class Chunk:
    idx: int
    path: Path
    loaded_tokens: Int[torch.Tensor, "doc seq"] | None = None
    dense_acts: Float[torch.Tensor, "doc seq d_dict"] | None = None
    sparse_acts: Float[torch.Tensor, "nnz 3"] | None = None
    _cfg: CachingConfig | None = None
    sparsify_batch_size: int = 100

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
        if self.sparse_acts is not None:
            return
        assert self.dense_acts is not None
        self.sparse_acts = self.dense_acts.to_sparse_coo()
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
        assert self.loaded_tokens is not None
        assert not self.tokens_path.exists()
        save_file({"tokens": self.loaded_tokens}, self.tokens_path)

    def load_sparse(self):
        self.sparse_acts = self.read_sparse()

    def load_dense(self):
        self.dense_acts = self.read_dense()

    def load_tokens(self):
        self.loaded_tokens = self.read_tokens()

    def read_sparse(self):
        if self.sparse_acts is not None:
            return self.sparse_acts
        return load_sparse_tensor(self.sparse_path)

    def read_dense(self):
        if self.dense_acts is not None:
            return self.dense_acts
        return load_file(self.dense_path)["acts"]

    def read_tokens(self):
        if self.loaded_tokens is not None:
            return self.loaded_tokens
        return load_file(self.tokens_path)["tokens"]

    @classmethod
    def load_from_dir(
        cls, path, index, load_sparse_only: bool = False, lazy=False
    ) -> "Chunk":
        inst = cls(path=path, idx=index)
        if lazy:
            return inst
        inst.load(load_sparse_only=load_sparse_only)
        return inst

    @classmethod
    def chunks_from_dir_iter(cls, path, load_sparse_only: bool = False, lazy=False):
        i = 0
        while True:
            try:
                chunk = cls.load_from_dir(path, i, load_sparse_only, lazy=lazy)
                if not chunk.exists:
                    break
                yield chunk
                i += 1
            except:
                break

    @classmethod
    def load_chunks_from_dir(cls, path, load_sparse_only: bool = False, lazy=False):
        return list(cls.chunks_from_dir_iter(path, load_sparse_only, lazy=lazy))

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
        # if not self.default_sparse:
        #     if self.dense_acts is None:
        #         return self.read_dense()
        #     return self.dense_acts
