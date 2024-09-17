from attrs import define, field
from saeco.evaluation.filtered_evaluation import Filter, FilteredTensor
from safetensors.torch import load_file, save_file

from .chunk import Chunk
from .sparse_safetensors import load_sparse_tensor, save_sparse_tensor


@define
class FilteredChunk(Chunk):
    # acts: FilteredTensor
    filter: Filter = None

    def read_sparse(self):
        if self.sparse_acts is not None:
            return self.sparse_acts
        loaded = load_sparse_tensor(self.sparse_path)
        if self.filter is None:
            raise ValueError("Filter is None")
        return self.filter.mask(loaded, chunk=self)

    def read_dense(self):
        if self.dense_acts is not None:
            return self.dense_acts
        loaded = load_file(self.dense_path)["acts"]
        if self.filter is None:
            raise ValueError("Filter is None")
        return self.filter.mask(loaded, chunk=self)

    def read_tokens(self):
        if self.loaded_tokens is not None:
            return self.loaded_tokens
        loaded = load_file(self.tokens_path)["tokens"]
        if self.filter is None:
            raise ValueError("Filter is None")
        return self.filter.mask(loaded, chunk=self)

    @classmethod
    def load_from_dir(
        cls, filter, path, index, load_sparse_only: bool = False, lazy=False
    ) -> "FilteredChunk":
        inst = cls(filter=filter, path=path, idx=index)
        if lazy:
            return inst
        inst.load(load_sparse_only=load_sparse_only)
        return inst

    @classmethod
    def chunks_from_dir_iter(
        cls, filter: Filter, path, load_sparse_only: bool = False, lazy=False
    ):
        i = 0
        while True:
            try:
                chunk = cls.load_from_dir(
                    filter=filter,
                    path=path,
                    index=i,
                    load_sparse_only=load_sparse_only,
                    lazy=lazy,
                )
                if not chunk.exists:
                    break
                yield chunk
                i += 1
            except:
                break

    @classmethod
    def load_chunks_from_dir(
        cls, filter: Filter, path, load_sparse_only: bool = False, lazy=False
    ) -> list["FilteredChunk"]:
        return list(
            cls.chunks_from_dir_iter(
                filter=filter,
                path=path,
                load_sparse_only=load_sparse_only,
                lazy=lazy,
            )
        )
