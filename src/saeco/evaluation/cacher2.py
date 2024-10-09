from pathlib import Path
from typing import Any, Callable, Iterator

import einops
import torch

import tqdm
from attrs import define, field
from jaxtyping import Float, Int
from torch import Tensor

from saeco.data import ActsData, DataConfig
from saeco.data.tokens_data import PermutedDocs

from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.evaluation.storage.chunk import Chunk
from saeco.evaluation.storage.sparse_growing_disk_tensor import SparseGrowingDiskTensor
from saeco.trainer import Trainable, TrainingRunner


def to_token_chunk_yielder(tokens: Tensor, chunk_size: int):
    for i in range(0, tokens.shape[0], chunk_size):
        yield tokens[i : i + chunk_size]


@define
class ActsCacher:
    cfg: CachingConfig
    tokens_source: Iterator
    model_context: TrainingRunner
    acts_data: ActsData
    seq_len: int
    feature_tensors: list[SparseGrowingDiskTensor] | None = field(
        init=False, default=None
    )
    chunk_counter: int = 0

    @classmethod
    def from_cache_and_runner(
        cls,
        caching_config: CachingConfig,
        model_context: TrainingRunner,
        split="val",
    ):
        pd = PermutedDocs(
            model_context.cfg.train_cfg.data_cfg,
            split=model_context.cfg.train_cfg.data_cfg.getsplit(split),
        )
        cfg = caching_config
        if cfg.exclude_bos_from_storage is None:
            cfg.exclude_bos_from_storage = (
                model_context.cfg.train_cfg.data_cfg.model_cfg.acts_cfg.excl_first
            )
        inst = cls(
            cfg=cfg,
            tokens_source=pd.iter_docs_and_columns(
                batch_size=caching_config.docs_per_chunk,
                columns=caching_config.metadatas_from_src_column_names,
            ),
            model_context=model_context,
            acts_data=model_context.cfg.train_cfg.data_cfg.acts_data(),
            seq_len=model_context.cfg.train_cfg.data_cfg.seq_len,
        )
        return inst

    def store_acts(self):
        metadata_chunks = self.store(self.cfg.num_chunks)
        (self.path / CachingConfig.STANDARD_FILE_NAME).write_text(
            self.cfg.model_dump_json()
        )
        return metadata_chunks

    @torch.inference_mode()
    def get_llm_acts(self, tokens: Int[Tensor, "doc seq"]):
        return self.acts_data.to_acts(
            tokens,
            llm_batch_size=self.cfg.llm_batch_size
            or self.cfg.documents_per_micro_batch,
            rearrange=False,
            force_not_skip_padding=True,
            skip_exclude=True,
        )

    @property
    def path(self):
        root = Path.home() / "workspace" / "cached_sae_acts"
        path = root / self.cfg.dirname
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def d_dict(self) -> int:
        return self.model_context.cfg.init_cfg.d_dict

    @property
    def sae_model(self):
        return self.model_context.trainable

    # @feature_tensors.default
    def _feature_tensors_initializer(self):
        if (
            self.cfg.store_feature_tensors
            or self.cfg.deferred_blocked_store_feats_block_size
        ):
            return [
                SparseGrowingDiskTensor.create(
                    self.path / "features" / f"feature{i}", shape=[None, self.seq_len]
                )
                for i in range(self.d_dict)
            ]
        return None

    def store(self, n_chunks=None):
        metadata_chunks = []
        self.feature_tensors = self._feature_tensors_initializer()
        assert self.cfg.store_dense or self.cfg.store_sparse
        for chunk_id, (chunk, metadata_columns) in tqdm.tqdm(
            enumerate(self.chunk_generator()), total=self.cfg.num_chunks
        ):
            if n_chunks is not None and chunk_id >= n_chunks:
                break
            chunk.save_tokens()
            if self.cfg.store_sparse:
                chunk.sparsify()
                chunk.save_sparse()
            if self.cfg.store_dense:
                chunk.save_dense()
            if self.cfg.store_feature_tensors:
                if not self.cfg.eager_sparse_generation:
                    print("Storing feature tensors")
                    for i, feat_acts in enumerate(chunk._dense_acts.split(1, dim=2)):
                        self.feature_tensors[i].append(feat_acts.squeeze(-1))
                else:
                    print("Storing feature tensors")

                    ff_sparse_acts = chunk._sparse_acts.transpose(0, 2).transpose(1, 2)
                    for i in range(self.d_dict):
                        self.feature_tensors[i].append(ff_sparse_acts[i])

            del chunk
            print(f"Stored chunk {chunk_id}")
            metadata_chunks.append(metadata_columns)

        if self.cfg.deferred_blocked_store_feats_block_size:
            B = 10
            K = 1
            features_batch_size = (
                self.d_dict // self.cfg.deferred_blocked_store_feats_block_size
            )
            prog = tqdm.tqdm(total=self.cfg.num_chunks)
            for i in range(0, self.d_dict, features_batch_size):
                batch = self.feature_tensors[i : i + features_batch_size]
                chunks = Chunk.load_chunks_from_dir(self.path, lazy=True)
                for j in range(0, len(chunks), B):
                    acts = None
                    for ci, chunk in enumerate(chunks[j : j + B]):
                        spacts = chunk.read_sparse_raw().cuda()
                        if acts is None:
                            acts = torch.zeros(
                                self.cfg.docs_per_chunk * B,
                                self.seq_len,
                                features_batch_size,
                                device="cuda",
                                dtype=spacts.dtype,
                            )
                        mask = (spacts.indices()[2] >= i) & (
                            spacts.indices()[2] < i + features_batch_size
                        )
                        ids = spacts.indices()[:, mask].clone()
                        ids[2] -= i
                        vals = spacts.values()[mask]
                        acts[
                            ci
                            * self.cfg.docs_per_chunk : (chunk.idx + 1)
                            * self.cfg.docs_per_chunk
                        ][ids.unbind()] = vals
                    for k, feat_acts in enumerate(batch):
                        feat_acts.append(acts[:, :, k])
                        # spa = acts.index_select(
                        #     dim=2,
                        #     index=torch.tensor(
                        #         [i + k], dtype=torch.long, device=acts.device
                        #     ),
                        # ).coalesce()
                        # spa = torch.sparse_coo_tensor(
                        #     indices=spa.indices()[:2],
                        #     values=spa.values(),
                        #     size=spa.shape[:2],
                        # )
                        # feat_acts.append(spa)

                    prog.update()

                print(f"Stored features {i} to {i + features_batch_size}")
            prog.close()
        if self.feature_tensors:
            for ft in self.feature_tensors:
                ft.finalize()
        return metadata_chunks

    def chunk_generator(self):
        for tokens, columns in self.tokens_source:
            if self.cfg.eager_sparse_generation:
                chunk = Chunk(
                    idx=self.chunk_counter,
                    path=self.path,
                    loaded_tokens=tokens,
                    sparse_acts=self.batched_tokens_to_sae_acts(
                        tokens, sparse_eager=True
                    ),
                )

            else:
                chunk = Chunk(
                    idx=self.chunk_counter,
                    path=self.path,
                    loaded_tokens=tokens,
                    dense_acts=self.batched_tokens_to_sae_acts(tokens),
                )
            self.chunk_counter += 1
            yield chunk, columns

    @torch.inference_mode()
    def batched_tokens_to_sae_acts(self, tokens: Tensor, sparse_eager: bool = False):
        # this could be optimized more by having separate batch sizes for llm and sae
        sae_acts = []
        for i in range(0, tokens.shape[0], self.cfg.documents_per_micro_batch):
            batch_tokens = tokens[i : i + self.cfg.documents_per_micro_batch]
            batch_subj_acts = self.get_llm_acts(batch_tokens)
            batch_sae_acts = self.get_sae_acts(batch_subj_acts)
            del batch_subj_acts
            # if sparse_eager and self.cfg.store_feature_tensors:
            #     for i, feat_acts in enumerate(batch_sae_acts.split(1, dim=2)):
            #         self.feature_tensors[i].append(feat_acts.squeeze(-1))
            if sparse_eager:
                sae_acts.append(batch_sae_acts.to_sparse_coo())
            else:
                sae_acts.append(batch_sae_acts)

        if sparse_eager:
            return torch.cat(sae_acts, dim=0).coalesce()
        return torch.cat(sae_acts, dim=0)

    def get_sae_acts(self, subj_acts) -> Tensor:
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
