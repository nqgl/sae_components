from pydantic import BaseModel
from typing import ClassVar


class CachingConfig(BaseModel):
    docs_per_chunk: int = 100
    num_chunks: int = 30
    store_sparse: bool = True
    store_dense: bool = False
    documents_per_micro_batch: int = 16
    llm_batch_size: int | None = None
    dirname: str = "test"
    store_feature_tensors: bool = True
    STANDARD_FILE_NAME: ClassVar = "cache_config.json"
