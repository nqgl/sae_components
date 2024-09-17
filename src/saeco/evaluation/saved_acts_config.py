from pydantic import BaseModel
from typing import ClassVar
from pathlib import Path


class CachingConfig(BaseModel):
    model_name: str | None = None
    docs_per_chunk: int = 100
    num_chunks: int = 30
    store_sparse: bool = True
    store_dense: bool = False
    documents_per_micro_batch: int = 16
    llm_batch_size: int | None = None
    dirname: str = "test"
    store_feature_tensors: bool = True
    eager_sparse_generation: bool = False
    exclude_bos_from_storage: bool | None = None
    STANDARD_FILE_NAME: ClassVar = "cache_config.json"

    @property
    def num_docs(self):
        return self.docs_per_chunk * self.num_chunks

    @property
    def path(self):
        return Path.home() / "workspace" / "cached_sae_acts" / self.dirname
