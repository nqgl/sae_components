from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path

import einops
import torch
import tqdm
from attrs import define
from jaxtyping import Int
from torch import Tensor

from saeco.architecture.architecture import Architecture
from saeco.data.dict_batch.dict_batch import DictBatch
from saeco.data.storage.sparse_growing_disk_tensor import SparseGrowingDiskTensor
from saeco.data.training_data import ActsDataCreator
from saeco.data.training_data.dictpiled_tokens_data import DictPiledTokensData
from saeco.data.training_data.sae_train_batch import SAETrainBatch
from saeco.data.training_data.tokens_data import PermutedDocs
from saeco.evaluation.storage.chunk import Chunk
from saeco.evaluation.storage.cache_config import CacheConfig
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig


@define(slots=True)
class ActsCacher:
    cfg: CacheConfig
    tokens_source: Iterator
    architecture: Architecture
    acts_data: ActsDataCreator
    seq_len: int
    chunk_counter: int = 0

    @classmethod
    def from_cache_and_runner[ArchCfgT: SweepableConfig](
        cls,
        caching_config: CacheConfig,
        architecture: Architecture[ArchCfgT],
        split: str = "val",
    ) -> "ActsCacher":
        cfg = caching_config

        if cfg.get_input_data_cls() == torch.Tensor:
            # This branch was intentionally incomplete in the original code.
            # Keep behavior (raise) but with a clearer message.
            pd = PermutedDocs(
                cfg=architecture.run_cfg.train_cfg.data_cfg,
                split=architecture.run_cfg.train_cfg.data_cfg.getsplit(split),
            )
            _ = pd  # kept for future; unused currently
            raise NotImplementedError(
                "Tensor input pipeline is not implemented here. "
                "Use DictBatch input or refactor the pipeline."
            )

        # DictBatch input pipeline
        override = architecture.run_cfg.train_cfg.data_cfg.override_dictpiler_path_str
        if override is None:
            raise ValueError("Expected dictpiler override for DictBatch input pipeline")

        tokens_data = architecture.run_cfg.train_cfg.data_cfg.tokens_data(split)
        if not isinstance(tokens_data, DictPiledTokensData):
            raise TypeError("Expected DictPiledTokensData for DictBatch pipeline")

        def gen_with_columns():
            for tokens in tokens_data.piler.batch_generator(batch_size=cfg.tokens_per_chunk):
                yield (
                    architecture.run_cfg.train_cfg.data_cfg.model_cfg.model_load_cfg.input_data_transform(tokens),
                    {col: tokens[col] for col in cfg.metadatas_from_src_column_names},
                )

        tokens_source = gen_with_columns()

        if cfg.exclude_bos_from_storage is None:
            cfg.exclude_bos_from_storage = (
                architecture.run_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.excl_first
            )

        if architecture.run_cfg.train_cfg.data_cfg.seq_len is None:
            raise ValueError("seq_len must be set in run config")

        return cls(
            cfg=cfg,
            tokens_source=tokens_source,
            architecture=architecture,
            acts_data=architecture.run_cfg.train_cfg.data_cfg.acts_data_creator(),
            seq_len=architecture.run_cfg.train_cfg.data_cfg.seq_len,
        )

    def store_acts(self):
        metadata_chunks = self.store(self.cfg.num_chunks)
        (self.path / CacheConfig.STANDARD_FILE_NAME).write_text(self.cfg.model_dump_json())
        return metadata_chunks

    @property
    def path(self) -> Path:
        root = Path.home() / "workspace" / "cached_sae_acts"
        path = root / self.cfg.dirname
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def d_dict(self) -> int:
        return self.architecture.run_cfg.init_cfg.d_dict

    @property
    def sae_model(self):
        return self.architecture.trainable

    @cached_property
    def feature_tensors(self) -> list[SparseGrowingDiskTensor] | None:
        if not (self.cfg.store_feature_tensors or self.cfg.deferred_blocked_store_feats_block_size):
            return None

        feat_root = self.path / "features"
        feat_root.mkdir(parents=True, exist_ok=True)

        return [
            SparseGrowingDiskTensor.create(
                feat_root / f"feature{i}",
                shape=[0, self.seq_len],
            )
            for i in range(self.d_dict)
        ]

    def store(self, n_chunks: int | None = None):
        metadata_chunks: list[dict] = []

        if not (self.cfg.store_dense or self.cfg.store_sparse):
            raise ValueError("Must store dense or sparse acts")

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
                if self.feature_tensors is None:
                    raise RuntimeError("feature_tensors not initialized")

                if not self.cfg.eager_sparse_generation:
                    if chunk.dense_acts is None:
                        raise RuntimeError("Expected dense_acts for feature tensor storage")
                    for i, feat_acts in enumerate(chunk.dense_acts.split(1, dim=2)):
                        self.feature_tensors[i].append(feat_acts.squeeze(-1))
                else:
                    if chunk.sparse_acts is None:
                        raise RuntimeError("Expected sparse_acts for eager sparse generation")
                    ff_sparse_acts = chunk.sparse_acts.transpose(0, 2).transpose(1, 2)
                    for i in range(self.d_dict):
                        self.feature_tensors[i].append(ff_sparse_acts[i])

            del chunk
            metadata_chunks.append(metadata_columns)

        # Deferred blocked store path kept (as in original), but left unchanged structurally:
        if self.cfg.deferred_blocked_store_feats_block_size:
            if self.feature_tensors is None:
                raise RuntimeError("feature_tensors not initialized")

            B = self.cfg.deferred_blocked_store_feats_block_size
            features_batch_size = self.d_dict // B

            prog = tqdm.tqdm(total=self.cfg.num_chunks)
            for i in range(0, self.d_dict, features_batch_size):
                batch = self.feature_tensors[i : i + features_batch_size]
                chunks = Chunk[self.cfg.get_input_data_cls()].load_chunks_from_dir(self.path, lazy=True)

                for j in range(0, len(chunks), B):
                    acts = None
                    for ci, chunk in enumerate(chunks[j : j + B]):
                        spacts = chunk.read_sparse_raw().cuda()
                        if acts is None:
                            acts = torch.zeros(
                                self.cfg.tokens_per_chunk * B,
                                self.seq_len,
                                features_batch_size,
                                device="cuda",
                                dtype=spacts.dtype,
                            )
                        mask = (spacts.indices()[2] >= i) & (spacts.indices()[2] < i + features_batch_size)
                        ids = spacts.indices()[:, mask].clone()
                        ids[2] -= i
                        vals = spacts.values()[mask]
                        acts[
                            ci * self.cfg.tokens_per_chunk : (chunk.idx + 1) * self.cfg.tokens_per_chunk
                        ][ids.unbind()] = vals

                    for k, feat_acts in enumerate(batch):
                        if acts is None:
                            raise RuntimeError("acts buffer missing")
                        feat_acts.append(acts[:, :, k])

                    prog.update()

                for spgdt in batch:
                    spgdt.finalize()

            prog.close()

        if self.feature_tensors:
            for ft in self.feature_tensors:
                if not ft.finalized:
                    raise RuntimeError(f"Feature tensor {ft} is not finalized")

        return metadata_chunks

    def chunk_generator(self):
        for tokens, columns in self.tokens_source:
            if self.cfg.eager_sparse_generation:
                chunk = Chunk(
                    idx=self.chunk_counter,
                    path=self.path,
                    loaded_input_data=tokens,
                    sparse_acts=self.batched_tokens_to_sae_acts(tokens, sparse_eager=True),
                )
            else:
                chunk = Chunk(
                    idx=self.chunk_counter,
                    path=self.path,
                    loaded_input_data=tokens,
                    dense_acts=self.batched_tokens_to_sae_acts(tokens),
                )

            self.chunk_counter += 1
            yield chunk, columns

    @torch.inference_mode()
    def batched_tokens_to_sae_acts(self, tokens: Tensor | DictBatch, *, sparse_eager: bool = False) -> Tensor:
        sae_acts: list[Tensor] = []
        batch_size = tokens.batch_size if isinstance(tokens, DictBatch) else tokens.shape[0]

        for i in range(0, batch_size, self.cfg.documents_per_micro_batch):
            batch_tokens = tokens[i : i + self.cfg.documents_per_micro_batch]
            subj_acts = self.get_llm_acts(batch_tokens)
            sae = self.get_sae_acts(subj_acts)
            del subj_acts
            sae_acts.append(sae.to_sparse_coo() if sparse_eager else sae)

        return torch.cat(sae_acts, dim=0).coalesce() if sparse_eager else torch.cat(sae_acts, dim=0)

    @torch.inference_mode()
    def get_llm_acts(self, tokens: Int[Tensor, "doc seq"] | DictBatch) -> DictBatch:
        return self.acts_data._to_acts_unprocessed_inputs(
            tokens,
            llm_batch_size=self.cfg.llm_batch_size or self.cfg.documents_per_micro_batch,
            rearrange=False,
            force_not_skip_padding=True,
            skip_exclude=True,
        )

    def get_sae_acts(self, subj_acts: DictBatch) -> Tensor:
        subj_acts_flat = subj_acts.einops_rearrange("doc seq d_dict -> (doc seq) d_dict")
        batch = SAETrainBatch(
            **subj_acts_flat,
            input_sites=self.architecture.run_cfg.train_cfg.input_sites
            or self.architecture.run_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.sites,
            target_sites=self.architecture.run_cfg.train_cfg.target_sites,
        )

        shapes = [t.shape for t in subj_acts.values()]
        if not all(shapes[0] == shape for shape in shapes):
            raise ValueError("All subj_acts tensors must share the same shape")

        ndoc, seq_len, d_dict = shapes[0]
        sae_acts_flat = self.sae_model.get_acts(batch.input)

        sae_acts = einops.rearrange(
            sae_acts_flat, "(doc seq) d_dict -> doc seq d_dict", doc=ndoc, seq=seq_len
        )

        if self.cfg.exclude_bos_from_storage:
            sae_acts[:, 0] = 0
        return sae_acts