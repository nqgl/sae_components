from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from paramsight import get_resolved_typevars_for_base, takes_alias
from pydantic import BaseModel
from torch import Tensor

from saeco.data.dict_batch import DictBatch


class CacheConfig[InputDataT: Tensor | DictBatch](BaseModel):
    model_path_str: str | None = None
    averaged_model_weights: bool = False

    docs_per_chunk: int = 100
    num_chunks: int = 30

    store_sparse: bool = True
    store_dense: bool = False
    src_piler_num_epochs: int | None = 1

    documents_per_micro_batch: int = 16
    llm_batch_size: int | None = None

    dirname: str = "test"

    store_feature_tensors: bool = True
    eager_sparse_generation: bool = True
    exclude_bos_from_storage: bool | None = None

    deferred_blocked_store_feats_block_size: int | None = 10
    last_chunk_num_docs: int | None = None

    STANDARD_FILE_NAME: ClassVar[str] = "cache_config.json"
    metadatas_from_src_column_names: list[str] = []

    @takes_alias
    @classmethod
    def get_input_data_cls(cls) -> type[InputDataT]:
        return get_resolved_typevars_for_base(cls, CacheConfig)[0]  # type: ignore

    @property
    def model_path(self) -> Path | None:
        return None if self.model_path_str is None else Path(self.model_path_str)

    @model_path.setter
    def model_path(self, value: str | Path):
        self.model_path_str = str(value)

    @property
    def num_docs(self) -> int:
        return self.docs_per_chunk * self.num_chunks

    @property
    def path(self) -> Path:
        return Path.home() / "workspace" / "cached_sae_acts" / self.dirname
