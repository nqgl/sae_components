from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel
from saeco.architecture.arch_reload_info import ArchStoragePaths


class CachingConfig(BaseModel):
    model_name: str | None = None
    model_path: ArchStoragePaths | None = None
    docs_per_chunk: int = 100
    num_chunks: int = 30
    store_sparse: bool = True
    store_dense: bool = False
    documents_per_micro_batch: int = 16
    llm_batch_size: int | None = None
    dirname: str = "test"
    store_feature_tensors: bool = True
    eager_sparse_generation: bool = True
    exclude_bos_from_storage: bool | None = None
    deferred_blocked_store_feats_block_size: int = 10
    STANDARD_FILE_NAME: ClassVar = "cache_config.json"
    metadatas_from_src_column_names: list[str] = []

    @property
    def num_docs(self):
        return self.docs_per_chunk * self.num_chunks

    @property
    def path(self):
        return Path.home() / "workspace" / "cached_sae_acts" / self.dirname
