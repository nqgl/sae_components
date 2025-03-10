from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel
from saeco.architecture.arch_reload_info import ArchStoragePaths


class CachingConfig(BaseModel):
    model_path_str: str | None = None
    averaged_model_weights: bool = False
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
    def model_path(self) -> Path | None:
        if self.model_path_str is None:
            return None
        return Path(self.model_path_str)

    @model_path.setter
    def model_path(self, value: str | Path):
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            self.model_path_str = value
        else:
            raise ValueError("model_path must be a str or Path")

    @property
    def num_docs(self):
        return self.docs_per_chunk * self.num_chunks

    @property
    def path(self):
        return Path.home() / "workspace" / "cached_sae_acts" / self.dirname
