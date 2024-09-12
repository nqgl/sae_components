import torch
from jaxtyping import Int, Float
from saeco.data import DataConfig
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.evaluation.storage.chunk import Chunk
from saeco.trainer import Trainable, TrainingRunner
from typing import Any, Callable, Iterator
from pathlib import Path
from attrs import define, field
from torch import Tensor
import einops
from saeco.evaluation.storage.sparse_growing_disk_tensor import SparseGrowingDiskTensor


# storage_dir: Path = Path.home() / "workspace/data/caching"


def to_token_chunk_yielder(tokens: Tensor, chunk_size: int):
    for i in range(0, tokens.shape[0], chunk_size):
        yield tokens[i : i + chunk_size]


class ActsCacher:
    def __init__(
        self,
        caching_config: CachingConfig,
        model_context: TrainingRunner,
        tokens_source: None | DataConfig | Int[Tensor, "doc seq"] | Iterator = None,
        split="val",
    ):
        self.cfg = caching_config
        if tokens_source is None:
            tokens_source = model_context.cfg.train_cfg.data_cfg
        if isinstance(tokens_source, DataConfig):
            tokens_source = tokens_source.get_split_tokens(split)
        if isinstance(tokens_source, Tensor):
            tokens_source = to_token_chunk_yielder(
                tokens_source, caching_config.docs_per_chunk
            )
        assert isinstance(tokens_source, Iterator)
        self.tokens_source = tokens_source
        self.model_context = model_context
        self.acts_data = self.model_context.cfg.train_cfg.data_cfg.acts_data()
        if self.cfg.exclude_bos_from_storage is None:
            self.cfg.exclude_bos_from_storage = (
                self.model_context.cfg.train_cfg.data_cfg.model_cfg.acts_cfg.excl_first
            )

    def store_acts(self):
        seq_len = next(self.tokens_source).shape[1]
        cacher = Cacher(
            d_dict=self.model_context.cfg.init_cfg.d_dict,
            seq_len=seq_len,
            cfg=self.cfg,
            tokens_source=self.tokens_source,
            get_llm_acts=self.get_llm_acts,
            sae_model=self.model_context.trainable,
            path=self.path(),
        )
        cacher.store(self.cfg.num_chunks)
        (self.path() / CachingConfig.STANDARD_FILE_NAME).write_text(
            self.cfg.model_dump_json()
        )

    @torch.inference_mode()
    def get_llm_acts(self, tokens: Int[Tensor, "doc seq"]):
        return self.acts_data.to_acts(
            tokens,
            llm_batch_size=self.cfg.llm_batch_size
            or self.cfg.documents_per_micro_batch,
            rearrange=False,
            skip_exclude=True,
        )

    def path(self):
        root = Path.home() / "workspace" / "cached_sae_acts"
        path = root / self.cfg.dirname  # TODO
        path.mkdir(parents=True, exist_ok=True)
        return path


"""
external api:
    data[document_index, seq_position, feature_index]
under the hood:
    data[chunk, document_index, seq_position, feature_index]
"""
Tokens = Int[torch.Tensor, "doc seq"]
Acts = Float[torch.Tensor, "doc seq d_data"]


@define
class Cacher:
    d_dict: int
    seq_len: int
    cfg: CachingConfig
    tokens_source: Iterator[Tokens]
    get_llm_acts: Callable[[Tokens], Acts]
    sae_model: Trainable
    path: Path
    feature_tensors: list[SparseGrowingDiskTensor] | None = field(init=False)
    chunk_counter: int = 0

    @feature_tensors.default
    def _feature_tensors_initializer(self):
        if self.cfg.store_feature_tensors:
            return [
                SparseGrowingDiskTensor.create(
                    self.path / "features" / f"feature{i}", shape=[None, self.seq_len]
                )
                for i in range(self.d_dict)
            ]
        return None

    def store(self, n_chunks=None):
        assert self.cfg.store_dense or self.cfg.store_sparse
        for chunk_id, chunk in enumerate(self.chunk_generator()):
            if n_chunks is not None and chunk_id >= n_chunks:
                break
            chunk.save_tokens()
            if self.cfg.store_sparse:
                chunk.sparsify()
                chunk.save_sparse()
            if self.cfg.store_dense:
                chunk.save_dense()
            if self.cfg.store_feature_tensors:
                print("Storing feature tensors")
                for i, feat_acts in enumerate(chunk.dense_acts.split(1, dim=2)):
                    self.feature_tensors[i].append(feat_acts.squeeze(-1))
            del chunk
            print(f"Stored chunk {chunk_id}")
        for ft in self.feature_tensors:
            ft.finalize()

    def chunk_generator(self):
        for tokens in self.tokens_source:
            chunk = Chunk(
                idx=self.chunk_counter,
                path=self.path,
                loaded_tokens=tokens,
                dense_acts=self.batched_tokens_to_sae_acts(tokens),
            )
            self.chunk_counter += 1
            yield chunk

    @torch.inference_mode()
    def batched_tokens_to_sae_acts(self, tokens: Tensor):
        # this could be optimized more by having separate batch sizes for llm and sae
        sae_acts = []
        for i in range(0, tokens.shape[0], self.cfg.documents_per_micro_batch):
            batch_tokens = tokens[i : i + self.cfg.documents_per_micro_batch]
            batch_subj_acts = self.get_llm_acts(batch_tokens)
            batch_sae_acts = self.get_sae_acts(batch_subj_acts)
            del batch_subj_acts
            sae_acts.append(batch_sae_acts)
        return torch.cat(sae_acts, dim=0)

    def get_sae_acts(self, subj_acts):
        ndoc, seq_len, d_dict = subj_acts.shape
        subj_acts_flat = einops.rearrange(
            subj_acts, "doc seq d_dict -> (doc seq) d_dict"
        )
        sae_acts_flat = self.sae_model.get_acts(subj_acts_flat)
        sae_acts = einops.rearrange(
            sae_acts_flat, "(doc seq) d_dict -> doc seq d_dict", doc=ndoc, seq=seq_len
        )
        if self.cfg.exclude_bos_from_storage:
            sae_acts[:, 0] = 0
        return sae_acts


class FeaturesActivatedCached:
    indices: Int[torch.Tensor, "nnz 2"]  # nnz, spchunk, i
