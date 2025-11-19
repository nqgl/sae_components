import shelve
from collections.abc import Generator
from functools import cached_property
from pathlib import Path
from typing import Union

import einops
import torch
import tqdm
from attrs import define, field
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from saeco.architecture.architecture import Architecture
from saeco.data.config.locations import DATA_DIRS
from saeco.evaluation.BMStorShelf import BMStorShelf
from saeco.evaluation.eval_components.coacts import Coactivity
from saeco.evaluation.eval_components.enrichment import Enrichment
from saeco.evaluation.eval_components.patching import Patching
from saeco.evaluation.MetadataBuilder import FilteredBuilder, MetadataBuilder
from saeco.trainer import RunConfig

from .cached_artifacts import CachedCalls
from .cacher import ActsCacher, CachingConfig
from .eval_components.family_generation import FamilyGenerator
from .eval_components.family_ops import FamilyOps
from .fastapi_models import (
    Feature,
)
from .fastapi_models.families_draft import (
    FamilyRef,
)
from .filtered import FilteredTensor
from .named_filter import NamedFilter
from .saved_acts import SavedActs
from .storage.chunk import Chunk
from .storage.stored_metadata import Artifacts, Filters, Metadatas


@define
class Evaluation(FamilyGenerator, FamilyOps, Enrichment, Patching, Coactivity):
    model_path: Path
    architecture: Architecture = field(repr=False)
    averaged_model_weights: bool = field(default=False, repr=False)
    saved_acts: SavedActs | None = field(default=None, repr=False)
    _filter: NamedFilter | None = field(default=None)
    tokenizer: PreTrainedTokenizerFast = field()
    _root: Union["Evaluation", None] = field(default=None, repr=False)

    @tokenizer.default
    def _tokenizer_default(self):
        return AutoTokenizer.from_pretrained(
            self.sae_cfg.train_cfg.data_cfg.model_cfg.model_name,
            cache_dir=DATA_DIRS.CACHE_DIR,
        )

    @cached_property
    def feature_labels(self):
        return shelve.open(str(self.root.path / "feature_labels"))

    @cached_property
    def family_labels(self):
        return shelve.open(str(self.path / "family_labels"))

    @cached_property
    def bmstore(self):
        return BMStorShelf.from_path(self.path)

    @property
    def root(self):
        if self._root is None:
            return self
        return self._root

    @property
    def cuda(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @property
    def docs(self):
        return self.saved_acts.tokens

    @property
    def docstrs(self) -> "StrDocs":
        return StrDocs(self)

    @property
    def acts(self):
        return self.saved_acts.acts

    @property
    def nnsight_model(self):
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.model

    @classmethod
    def from_cache_name(cls, name: Path | str):
        if isinstance(name, str):
            name = Path(name)
            if not name.exists():
                name = Path.home() / "workspace" / "cached_sae_acts" / name
        saved = SavedActs.from_path(name)
        inst = cls.from_model_path(
            saved.cfg.model_path, averaged_weights=saved.cfg.averaged_model_weights
        )
        inst.saved_acts = saved
        return inst

    @classmethod
    def from_model_path(cls, path: Path, averaged_weights: bool = False):
        path = path if isinstance(path, Path) else Path(path)
        arch = Architecture.load(
            path, load_weights=True, averaged_weights=averaged_weights
        )
        inst = cls(
            architecture=arch, model_path=path, averaged_model_weights=averaged_weights
        )

        return inst

    def __attrs_post_init__(self):
        if self.saved_acts is not None:
            if self.saved_acts.data_filter is not self._filter:
                raise ValueError("Filter mismatch between Evaluation and storage")
        else:
            assert self._filter is None

    def _apply_filter(self, filter: NamedFilter | Tensor):
        if isinstance(filter, Tensor):
            filter = NamedFilter(filter=filter, filter_name=None)
        if self._filter is not None:
            raise ValueError(
                "Filter already set, create filtered from the root Evaluation"
            )
        return Evaluation(
            model_path=self.model_path,
            architecture=self.architecture,
            saved_acts=self.saved_acts.filtered(filter),
            filter=filter,
            root=self,
            tokenizer=self.tokenizer,
        )

    def open_filtered(self, filter_name: str):
        return self._apply_filter(self.filters[filter_name])

    def _make_metadata_builder_iter(  ###
        self, dtype, device, item_size=[]
    ) -> Generator[Chunk, FilteredTensor, Tensor]:
        assert self._filter is None
        new_tensor = torch.zeros(
            self.cache_cfg.num_docs, *item_size, dtype=dtype, device=device
        )

        for chunk in self.saved_acts.chunks:
            value = yield chunk
            yield
            assert isinstance(value, FilteredTensor | Tensor)
            if isinstance(value, Tensor):
                value = chunk._to_filtered(value)
            value.filter.writeat(new_tensor, value.value)
        return new_tensor

    def metadata_builder(self, dtype, device, item_size=[]) -> "MetadataBuilder":
        return MetadataBuilder(
            self.saved_acts.chunks,
            dtype=dtype,
            device=device,
            shape=[self.cache_cfg.num_docs, *item_size],
        )

    def filtered_builder(self, dtype, device, item_size=[]) -> "FilteredBuilder":
        return FilteredBuilder(
            self.saved_acts.chunks,
            dtype=dtype,
            device=device,
            shape=[self.cache_cfg.num_docs, *item_size],
            filter=self._filter,
        )

    @property
    def path(self):
        if self.saved_acts is None:
            raise ValueError("cache_name must be set")
        if self._filter is None:
            return self.cache_cfg.path
        return self._filter.filtered_dir(self.cache_cfg.path)

    @cached_property
    def cached_call(self) -> Union[CachedCalls, "Evaluation"]:
        return CachedCalls(self)

    @cached_property
    def metadatas(self) -> Metadatas:
        if self._filter is not None:
            raise NotImplementedError(
                "Getting metadatas from a filtered Evaluation is TODO and pending some design choices."
            )
        return Metadatas(self.path, cached_config=self.cache_cfg)

    @property
    def _root_metadatas(self):
        if self._root is None:
            return self.metadatas
        return self._root.metadatas

    @cached_property
    def artifacts(self) -> Artifacts:
        return Artifacts(self.path, cached_config=self.cache_cfg)

    @cached_property
    def filters(self) -> Filters:
        if self._filter is not None:
            raise ValueError(
                "Cannot access filters from a filtered evaluation. If this could be useful though, let me (Glen) know."
            )

        return Filters(self.path, cached_config=self.cache_cfg)

    @property
    def d_dict(self):
        return self.architecture.run_cfg.init_cfg.d_dict

    @property
    def sae_cfg(self) -> RunConfig:
        return self.architecture.run_cfg

    @property
    def cache_cfg(self) -> CachingConfig:
        return self.saved_acts.cfg

    @property
    def features(self):
        return self.saved_acts.features

    @property
    def sae(self):
        return self.architecture.trainable

    @property
    def nnsight_site_name(self):
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.site

    @property
    def seq_len(self):
        return self.sae_cfg.train_cfg.data_cfg.seq_len

    @property
    def d_vocab(self):
        return self.tokenizer.vocab_size

    def store_acts(self, caching_cfg: CachingConfig, displace_existing=False):
        if caching_cfg.model_path is None:
            caching_cfg.model_path = self.model_path
        assert caching_cfg.model_path == self.model_path
        assert caching_cfg.averaged_model_weights == self.averaged_model_weights
        acts_cacher = ActsCacher.from_cache_and_runner(
            caching_config=caching_cfg, architecture=self.architecture
        )
        if acts_cacher.path.exists():
            if displace_existing:
                import time

                old = acts_cacher.path.parent / "old"
                old.mkdir(exist_ok=True, parents=True)
                acts_cacher.path.rename(
                    old / f"old_{time.time()}{acts_cacher.path.name}"
                )
            else:
                raise FileExistsError(
                    f"{acts_cacher.path} already exists. Set displace_existing=True to move existing files."
                )

        metadata_chunks = acts_cacher.store_acts()
        self.saved_acts = SavedActs.from_path(acts_cacher.path)
        metadata_builders = {
            name: self.metadata_builder(torch.long, "cpu")
            for name in self.cache_cfg.metadatas_from_src_column_names
        }
        for mchunk in metadata_chunks:
            for name in mchunk:
                metadata_builders[name].takestrl(mchunk[name])
        for name, builder in metadata_builders.items():
            self.metadatas[name] = builder.value
            self.metadatas.set_str_translator(name, builder.unique_labels)

    def get_features(self, feature_ids):  ###
        return [self.features[fid] for fid in feature_ids]

    def get_feature(self, feature_id) -> FilteredTensor:  ###
        return self.features[feature_id]

    def filter_docs(self, docs_filter, only_return_selected=False, seq_level=False):
        ###
        if not only_return_selected:
            if seq_level:
                mask = torch.zeros(
                    self.cache_cfg.num_docs,
                    self.sae_cfg.train_cfg.data_cfg.seq_len,
                    dtype=torch.bool,
                )

            else:
                mask = torch.zeros(self.cache_cfg.num_docs, dtype=torch.bool)
        values = []
        for chunk in self.saved_acts.chunks:
            tokens = chunk.tokens.to(docs_filter.device)
            filt_docs = tokens.mask_by_other(
                docs_filter, return_ft=True, presliced=False
            )
            values.append(filt_docs.value)
            if not only_return_selected:
                filt_docs.filter.slice(mask)[:] = filt_docs.filter.mask
        values = torch.cat(values, dim=0)
        if only_return_selected:
            return values
        return FilteredTensor.from_value_and_mask(value=values, mask=mask)

    def filter_acts(self, docs_filter, only_return_selected=False):
        ###
        if not only_return_selected:
            mask = torch.zeros(self.cache_cfg.num_docs, dtype=torch.bool)
        values = []
        for chunk in self.saved_acts.chunks:
            acts = chunk.acts.to(docs_filter.device)
            filt = acts.filter.slice(docs_filter)
            if not filt.any():
                continue
            filt_docs = acts.mask_by_other(filt, return_ft=True, presliced=True)
            values.append(filt_docs.value)
            if not only_return_selected:
                filt_docs.filter.slice(mask)[:] = filt_docs.filter.mask
        values = torch.cat(values, dim=0)
        if only_return_selected:
            return values
        return FilteredTensor.from_value_and_mask(value=values, mask=mask)

    # def coactivations(self, doc_agg=None):
    #     sims = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
    #     coact_counts = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
    #     fa_sq_sum = torch.zeros(self.d_dict).to(self.cuda)
    #     for chunk in tqdm.tqdm(
    #         self.saved_acts.chunks, total=len(self.saved_acts.chunks)
    #     ):
    #         acts = chunk.acts.value.to(self.cuda).to_dense()
    #         assert acts.ndim == 3
    #         feature_activity = self.sequelize(
    #             acts, doc_agg=doc_agg
    #         )  # feat, (doc [seq])
    #         feature_bin = (feature_activity > 0).float()
    #         fa_sq_sum += feature_activity.pow(2).sum(-1)
    #         sims += feature_activity @ feature_activity.transpose(-2, -1)
    #         coact_counts += feature_bin @ feature_bin.transpose(-2, -1)
    #     norms = fa_sq_sum.sqrt()
    #     sims /= norms.unsqueeze(0)
    #     sims /= norms.unsqueeze(1)
    #     prod = sims.diag()[~sims.diag().isnan()].prod()
    #     assert prod < 1.001 and prod > 0.999
    #     return coact_counts, sims

    def get_feature_label(self, feature_id):
        return self.feature_labels.get(str(int(feature_id)))

    def set_feature_label(self, feature_id, label):
        self.feature_labels[str(int(feature_id))] = label

    def get_family_label(self, family):
        return self.family_labels.get(str((int(family.level), int(family.family_id))))

    def set_family_label(self, family: FamilyRef, label: str):
        self.family_labels[str((int(family.level), int(family.family_id)))] = label

    def get_feature(self, feat_id) -> Feature:
        return Feature(
            feature_id=int(feat_id),
            label=self.get_feature_label(feat_id),
        )

    def detokenize(self, tokens) -> list[str] | list[list[str]] | str:
        if isinstance(tokens, int):
            tokens = [tokens]
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        assert isinstance(tokens, Tensor), (
            "hmu if this assumption is wrong somewhere, easy fix"
        )
        if tokens.ndim == 0:
            return self.tokenizer._tokenizer.decode([tokens])
        if tokens.ndim == 1:
            return self.tokenizer._tokenizer.decode_batch(
                [[t] for t in tokens],
                skip_special_tokens=False,
            )

        lens = tokens.shape[1]
        flat = einops.rearrange(tokens, "doc seq -> (doc seq)").unsqueeze(-1).tolist()
        flatl = self.tokenizer._tokenizer.decode_batch(flat, skip_special_tokens=False)
        return [flatl[i : i + lens] for i in range(0, len(flatl), lens)]

    def seq_aggregated_chunks_yielder(self, seq_agg) -> Generator[FilteredTensor]:
        """
        seq_agg options: "mean", "max", "sum", "count", "any"
        - count: count number of non-zero activations in each doc
        - any: if feature has any non-zero activation in a doc

        """
        # move to saved acts?
        for chunk in tqdm.tqdm(self.saved_acts.chunks):
            acts = chunk.acts
            acts_inner = acts.value.to(self.cuda).to_dense()
            assert acts_inner.ndim == 3
            if seq_agg == "count":
                c_agg = (acts_inner > 0).sum(dim=1)
            elif seq_agg == "any":
                c_agg = (acts_inner > 0).any(dim=1)
            elif seq_agg == "max":
                c_agg = acts_inner.max(dim=1).values
            else:
                c_agg = getattr(acts_inner, seq_agg)(dim=1)
            yield acts.to_filtered_like_self(c_agg)

    @property
    def num_docs(self) -> int:
        if self._filter:
            return self._filter.filter.sum().item()
        return self.cache_cfg.num_docs

    def acts_avg_over_dataset(self, seq_agg="mean", docs_agg="mean"):
        """
        seq_agg options: "mean", "max", "sum", "count", "any"
        docs_agg options: "mean", "max", "sum"
        """
        results = torch.zeros(self.d_dict).to(self.cuda)

        for agg_chunk in self.seq_aggregated_chunks_yielder(seq_agg):
            if docs_agg == "max":
                results = (
                    torch.cat([results, agg_chunk.value.max(dim=0).values])
                    .max(dim=0)
                    .values
                )
            else:
                results += agg_chunk.value.sum(dim=0)
        if docs_agg == "mean":
            results /= self.num_docs

    @property
    def token_occurrence_count(self):
        return self.cached_call.count_token_occurrence()

    def top_activating_examples(self, feature_id: int, p=None, k=None):
        feature = self.features[feature_id]
        top = self._get_top_activating(feature.value, p=p, k=k)
        return feature.to_filtered_like_self(top)

    @staticmethod
    def _pk_to_k(p, k, quantity):
        if (p is None) == (k is None):
            raise ValueError("Exactly one of p and k must be set")
        if p is not None and not (0 < p <= 1):
            raise ValueError("p must be in (0, 1]")
        if k is None:
            k = int(quantity * p)
        if k <= 0:
            raise ValueError("k must be positive")
        return min(k, quantity)

    @staticmethod
    def _get_top_activating(feature: Tensor, p=None, k=None):
        k = Evaluation._pk_to_k(p, k, feature.shape[0])
        values = feature.values()
        if k >= values.shape[0]:
            k = values.shape[0]
        topk = values.topk(k)

        return torch.sparse_coo_tensor(
            feature.indices()[:, topk.indices],
            topk.values,
            feature.shape,
        )

    def seq_agg_feat(
        self, feature_id=None, feature=None, agg="max", docs_filter=True
    ) -> FilteredTensor:
        assert agg in ("max", "sum"), "Only max implemented currently"
        if (feature_id is None) == (feature is None):
            raise ValueError("Exactly one of feat_id and feature must be set")
        if feature is None:
            feature = self.features[feature_id]
        if docs_filter:
            feature = feature.filter_inactive_docs()
        if agg == "max":
            return feature.to_filtered_like_self(
                feature.value.to_dense().max(dim=1).values, ndim=1
            )
        elif agg == "sum":
            return feature.to_filtered_like_self(
                feature.value.to_dense().sum(dim=1), ndim=1
            )

    def top_activations_and_metadatas(
        self,
        feature: int | FilteredTensor,
        p: float = None,
        k: int = None,
        metadata_keys: list[str] = [],
        return_str_docs: bool = False,
        return_acts_sparse: bool = False,
        return_doc_indices: bool = True,
        str_metadatas: bool = False,
    ):
        if isinstance(feature, int):
            feature = self.features[feature]
        doc_acts = self.seq_agg_feat(feature=feature)
        k = Evaluation._pk_to_k(p, k, doc_acts.value.shape[0])
        topk = doc_acts.value.topk(k, sorted=True)
        top_outer_indices = doc_acts.externalize_indices(topk.indices.unsqueeze(0))
        doc_indices = top_outer_indices[0]
        acts = feature.index_select(doc_indices, dim=0)
        assert (acts.to_dense() == feature.to_dense()[top_outer_indices]).all()
        if return_acts_sparse:
            acts = acts.to_sparse_coo()
        docs = self.docstrs[doc_indices] if return_str_docs else self.docs[doc_indices]
        metadatas = {
            key: self._root_metadatas[key][doc_indices.cpu()] for key in metadata_keys
        }
        if str_metadatas:
            metadatas = self._root_metadatas.translate(metadatas)
        if return_doc_indices:
            return docs, acts, metadatas, doc_indices
        return docs, acts, metadatas

    def batched_top_activations_and_metadatas(
        self,
        features: list[int | FilteredTensor],
        p=None,
        k=None,
        metadata_keys=[],
        return_str_docs=False,
        return_acts_sparse=False,
        return_doc_indices=True,
        str_metadatas=False,
    ):
        return [
            self.top_activations_and_metadatas(
                feature,
                p,
                k,
                metadata_keys,
                return_str_docs,
                return_acts_sparse,
                return_doc_indices,
                str_metadatas,
            )
            for feature in features
        ]

    def getDAM(
        self,
        doc_indices,
        features: list[FilteredTensor],
        metadata_keys: list[str],
        return_str_docs: bool,
        str_metadatas: bool,
    ):
        acts = [f.index_select(doc_indices, dim=0) for f in features]
        docs = self.docstrs[doc_indices] if return_str_docs else self.docs[doc_indices]

        docs, metadatas = self.get_docs_and_metadatas(
            doc_indices,
            metadata_keys=metadata_keys,
            return_str_docs=return_str_docs,
            return_str_metadatas=str_metadatas,
        )

        return docs, acts, metadatas

    def get_docs_and_metadatas(
        self,
        doc_indices: Tensor,
        metadata_keys: list[str],
        return_str_docs: bool,
        return_str_metadatas: bool,
    ):
        docs = self.docstrs[doc_indices] if return_str_docs else self.docs[doc_indices]
        metadatas = {
            key: self._root_metadatas[key][doc_indices] for key in metadata_keys
        }
        if return_str_metadatas:
            metadatas = self._root_metadatas.translate(metadatas)
        return docs, metadatas

    def _metadata_unique_labels_and_counts_tensor(self, key):
        meta = self._root_metadatas[key]
        if self._filter is not None:
            meta = meta[self._filter.filter]
        assert meta.ndim == 1 and meta.dtype == torch.long
        labels, counts = meta.unique(return_counts=True)
        return torch.stack([labels, counts], dim=0)

    def count_token_occurrence(self):
        counts = torch.zeros(self.d_vocab, dtype=torch.long).to(self.cuda)
        for chunk in self.saved_acts.chunks:
            toks = chunk.tokens.value.to(self.cuda).flatten()
            counts.scatter_add_(
                0,
                toks,
                torch.ones(1, device=toks.device, dtype=torch.long).expand(
                    toks.shape[0]
                ),
            )
        return counts

    def num_active_docs_for_feature(self, feature_id):
        return self.cached_call._feature_num_active_docs()[feature_id].item()

    @property
    def seq_activation_counts(self):
        return self.cached_call._feature_num_active_tokens().cpu()

    @property
    def seq_activation_probs(self):
        return self.seq_activation_counts / (self.num_docs * self.seq_len)

    @property
    def doc_activation_counts(self):
        return self.cached_call._feature_num_active_docs().cpu()

    @property
    def doc_activation_probs(self):
        return self.doc_activation_counts / self.num_docs

    # @property
    # def mean_feature_activations(self)

    # def _feature_mean_activations(self):

    # def feature_activation_proportion_thresholds(self, p):

    def _feature_num_active_docs(self):
        activity = torch.zeros(self.d_dict, dtype=torch.long).to(self.cuda)
        for chunk in self.saved_acts.chunks:
            acts = chunk.acts.value.to(self.cuda).to_dense()
            activity += (acts > 0).any(dim=1).sum(dim=0)
        return activity

    def _feature_num_active_tokens(self):
        activity = torch.zeros(self.d_dict, dtype=torch.long).to(self.cuda)
        for chunk in self.saved_acts.chunks:
            acts = chunk.acts.value.to(self.cuda).to_dense()
            activity += (acts > 0).sum(dim=1).sum(dim=0)
        return activity

    def get_metadata_intersection_filter_key(
        self, values: dict[str, str | list[str] | int | list[int]]
    ):
        val_list = list(values.items())
        val_list.sort()
        d = {}
        for k, v in val_list:
            if isinstance(v, int | str):
                v = [v]
            if isinstance(v, list) and isinstance(v[0], str):
                meta = self.metadatas.get(k)
                v = [meta.info.fromstr[x] for x in v]
            v = sorted(v)
            d[k] = tuple(v)
        key = str(d)

        if key not in self.filters:
            self.filters[key] = self._get_metadata_intersection_filter(d)
        return key

    def _get_metadata_intersection_filter(self, map: dict[str, list[int]]):
        filter = torch.ones(self.num_docs, dtype=torch.bool).to(self.cuda)
        for mdname, values in map.items():
            mdmask = torch.zeros(self.num_docs, dtype=torch.bool).to(self.cuda)
            meta = self.metadatas[mdname].to(self.cuda)
            for value in values:
                mdmask |= meta == value
            filter &= mdmask
        return filter


@define
class StrDocs:
    eval: Evaluation

    def __getitem__(self, idx):
        tokens = self.eval.docs[idx.cpu()]
        strs = self.eval.detokenize(tokens)
        assert len(strs) == tokens.shape[0] and len(strs[0]) == tokens.shape[1]
        return strs
