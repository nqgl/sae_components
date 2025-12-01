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
from saeco.evaluation.storage.saved_acts_config import CachingConfig
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig


def to_token_chunk_yielder(tokens: Tensor, chunk_size: int):
    for i in range(0, tokens.shape[0], chunk_size):
        yield tokens[i : i + chunk_size]


@define
class ActsCacher:
    cfg: CachingConfig
    tokens_source: Iterator
    architecture: Architecture
    acts_data: ActsDataCreator
    seq_len: int
    chunk_counter: int = 0

    @classmethod
    def from_cache_and_runner[ArchCfgT: SweepableConfig](
        cls,
        caching_config: CachingConfig,
        architecture: Architecture[ArchCfgT],
        split="val",
    ):
        if caching_config.get_input_data_cls() == torch.Tensor:
            assert (
                architecture.run_cfg.train_cfg.data_cfg.override_dictpiler_path_str
                is None
            )
            pd = PermutedDocs(
                cfg=architecture.run_cfg.train_cfg.data_cfg,
                split=architecture.run_cfg.train_cfg.data_cfg.getsplit(split),
            )
            tokens_source = pd.iter_docs_and_columns(
                batch_size=caching_config.docs_per_chunk,
                columns=caching_config.metadatas_from_src_column_names,
            )
        else:
            assert (
                architecture.run_cfg.train_cfg.data_cfg.override_dictpiler_path_str
                is not None
            )
            tokens_data = architecture.run_cfg.train_cfg.data_cfg.tokens_data(split)
            assert isinstance(tokens_data, DictPiledTokensData)

            def gen_with_columns():
                for tokens in tokens_data.piler.batch_generator(
                    batch_size=caching_config.docs_per_chunk
                ):
                    yield (
                        tokens,
                        {
                            col: tokens[col]
                            for col in caching_config.metadatas_from_src_column_names
                        },
                    )

            tokens_source = gen_with_columns()
        cfg = caching_config
        if cfg.exclude_bos_from_storage is None:
            cfg.exclude_bos_from_storage = (
                architecture.run_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.excl_first
            )
        assert architecture.run_cfg.train_cfg.data_cfg.seq_len is not None
        inst = cls(
            cfg=cfg,
            tokens_source=tokens_source,
            architecture=architecture,
            acts_data=architecture.run_cfg.train_cfg.data_cfg.acts_data_creator(),
            seq_len=architecture.run_cfg.train_cfg.data_cfg.seq_len,
        )
        return inst

    def store_acts(self):
        metadata_chunks = self.store(self.cfg.num_chunks)
        (self.path / CachingConfig.STANDARD_FILE_NAME).write_text(
            self.cfg.model_dump_json()
        )
        return metadata_chunks

    @property
    def path(self):
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
    def feature_tensors(self):
        if (
            self.cfg.store_feature_tensors
            or self.cfg.deferred_blocked_store_feats_block_size
        ):
            return [
                SparseGrowingDiskTensor.create(
                    self.path / "features" / f"feature{i}",
                    shape=[0, self.seq_len],
                    # TODO shape none fix and make path common
                )
                for i in range(self.d_dict)
            ]
        return None

    def store(self, n_chunks=None):
        metadata_chunks: list[dict] = []
        # self.feature_tensors = self._feature_tensors_initializer()
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
                assert self.feature_tensors is not None
                if not self.cfg.eager_sparse_generation:
                    print("Storing feature tensors")
                    for i, feat_acts in enumerate(chunk.dense_acts.split(1, dim=2)):
                        self.feature_tensors[i].append(feat_acts.squeeze(-1))
                else:
                    print("Storing feature tensors")

                    ff_sparse_acts = chunk.sparse_acts.transpose(0, 2).transpose(1, 2)
                    for i in range(self.d_dict):
                        self.feature_tensors[i].append(ff_sparse_acts[i])

            del chunk
            print(f"Stored chunk {chunk_id}")
            metadata_chunks.append(metadata_columns)

        if self.cfg.deferred_blocked_store_feats_block_size:
            assert self.feature_tensors is not None
            B = self.cfg.deferred_blocked_store_feats_block_size
            K = 1
            features_batch_size = (
                self.d_dict // self.cfg.deferred_blocked_store_feats_block_size
            )
            prog = tqdm.tqdm(total=self.cfg.num_chunks)
            for i in range(0, self.d_dict, features_batch_size):
                batch = self.feature_tensors[i : i + features_batch_size]
                chunks = Chunk[self.cfg.get_input_data_cls()].load_chunks_from_dir(
                    self.path, lazy=True
                )
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
                            ci * self.cfg.docs_per_chunk : (chunk.idx + 1)
                            * self.cfg.docs_per_chunk
                        ][ids.unbind()] = vals
                    for k, feat_acts in enumerate(batch):
                        assert acts is not None
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
                for spgdt in batch:
                    spgdt.finalize()
            prog.close()
        if self.feature_tensors:
            for ft in self.feature_tensors:
                assert ft.finalized, f"Feature tensor {ft} is not finalized"
        return metadata_chunks

    def chunk_generator(self):
        for tokens, columns in self.tokens_source:
            if self.cfg.eager_sparse_generation:
                chunk = Chunk(
                    idx=self.chunk_counter,
                    path=self.path,
                    loaded_input_data=tokens,
                    sparse_acts=self.batched_tokens_to_sae_acts(
                        tokens, sparse_eager=True
                    ),
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
    def batched_tokens_to_sae_acts(
        self, tokens: Tensor | DictBatch, sparse_eager: bool = False
    ):
        # this could be optimized more by having separate batch sizes for llm and sae
        sae_acts = []
        batch_size = (
            tokens.batch_size if isinstance(tokens, DictBatch) else tokens.shape[0]
        )
        for i in range(0, batch_size, self.cfg.documents_per_micro_batch):
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

    @torch.inference_mode()
    def get_llm_acts(self, tokens: Int[Tensor, "doc seq"] | DictBatch) -> DictBatch:
        return self.acts_data.to_acts(
            tokens,
            llm_batch_size=self.cfg.llm_batch_size
            or self.cfg.documents_per_micro_batch,
            rearrange=False,
            force_not_skip_padding=True,
            skip_exclude=True,
        )

    def get_sae_acts(self, subj_acts: DictBatch) -> Tensor:
        subj_acts_flat = subj_acts.einops_rearrange(
            "doc seq d_dict -> (doc seq) d_dict"
        )
        batch = SAETrainBatch(
            **subj_acts_flat,
            input_sites=self.architecture.run_cfg.train_cfg.input_sites
            or self.architecture.run_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.sites,
            target_sites=self.architecture.run_cfg.train_cfg.target_sites,
        )
        shapes = [t.shape for t in subj_acts.values()]
        assert all(shapes[0] == shape for shape in shapes)
        shape = shapes[0]
        ndoc, seq_len, d_dict = shape
        sae_acts_flat = self.sae_model.get_acts(
            batch.input
        )  # Or mb encode? optionally?
        sae_acts = einops.rearrange(
            sae_acts_flat, "(doc seq) d_dict -> doc seq d_dict", doc=ndoc, seq=seq_len
        )
        if self.cfg.exclude_bos_from_storage:
            sae_acts[:, 0] = 0
        return sae_acts
