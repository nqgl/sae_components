from __future__ import annotations

import shelve
from collections.abc import Generator
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import einops
import torch
import tqdm
from attrs import Factory, define, field
from paramsight import get_resolved_typevars_for_base, takes_alias
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from saeco.architecture.architecture import Architecture
from saeco.data.dict_batch import DictBatch
from saeco.evaluation.BMStorShelf import BMStorShelf
from saeco.evaluation.eval_components.coacts import Coactivity
from saeco.evaluation.eval_components.enrichment import Enrichment
from saeco.evaluation.eval_components.patching import Patching
from saeco.evaluation.model_interface import (
    ComlmEvalAdapter,
    LanguageModelEvalAdapter,
    ModelEvalAdapter,
)
from saeco.evaluation.return_objects import Feature, MetadataLabelCounts, TopActivations
from saeco.evaluation.storage.MetadataBuilder import FilteredBuilder, MetadataBuilder
from saeco.trainer import RunConfig

from .cached_artifacts import CachedCalls
from .eval_components.family_generation import FamilyGenerator
from .eval_components.family_ops import FamilyOps
from .fastapi_models import LabeledFeature
from .fastapi_models.families_draft import FamilyRef
from .filtered import FilteredTensor
from .named_filter import NamedFilter
from .storage.cacher import ActsCacher, CachingConfig
from .storage.saved_acts import SavedActs
from .storage.stored_metadata import Artifacts, Filters, Metadatas

if TYPE_CHECKING:
    from saeco.evaluation.features import Features
    from saeco.trainer.trainable import Trainable


def _extract_token_tensor(x: Tensor | DictBatch) -> Tensor:
    if isinstance(x, Tensor):
        return x
    # Try common keys.
    for key in ("tokens", "input_ids", "ids"):
        try:
            t = x[key]
            if isinstance(t, Tensor):
                return t
        except Exception:
            continue
    raise KeyError("Could not find token id tensor in DictBatch (expected one of: tokens, input_ids, ids)")


@define(slots=True)
class Evaluation[InputsT: torch.Tensor | DictBatch](
    FamilyGenerator, FamilyOps, Enrichment, Patching, Coactivity
):
    model_path: Path
    architecture: Architecture = field(repr=False)
    averaged_model_weights: bool = field(default=False, repr=False)
    saved_acts: SavedActs | None = field(default=None, repr=False)
    filter: NamedFilter | None = field(default=None)
    _root: "Evaluation | None" = field(default=None, repr=False, alias="root")

    def _tokenizer_default(self) -> PreTrainedTokenizerFast:
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.tokenizer

    tokenizer: PreTrainedTokenizerFast = field(
        default=Factory(_tokenizer_default, takes_self=True)
    )

    @takes_alias
    @classmethod
    def get_inputs_type(cls) -> type[InputsT]:
        return get_resolved_typevars_for_base(cls, Evaluation)[0]  # type: ignore

    def _model_adapter_default(self):
        model_kwargs = getattr(self.sae_cfg.train_cfg.data_cfg.model_cfg, "model_kwargs", {})  # type: ignore[attr-defined]
        try:
            from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig

            if isinstance(self.sae_cfg.train_cfg.data_cfg.model_cfg.model_load_cfg, ComlmModelConfig):  # type: ignore[attr-defined]
                return ComlmEvalAdapter(model_kwargs=model_kwargs)
        except Exception:
            pass
        return LanguageModelEvalAdapter(model_kwargs=model_kwargs)

    model_adapter: ModelEvalAdapter = field(
        default=Factory(_model_adapter_default, takes_self=True)
    )

    @cached_property
    def feature_labels(self):
        p = self.root.path
        p.mkdir(parents=True, exist_ok=True)
        return shelve.open(str(p / "feature_labels"))

    @cached_property
    def family_labels(self):
        p = self.path
        p.mkdir(parents=True, exist_ok=True)
        return shelve.open(str(p / "family_labels"))

    @cached_property
    def bmstore(self) -> BMStorShelf:
        return BMStorShelf.from_path(self.path)

    @property
    def root(self) -> "Evaluation":
        return self if self._root is None else self._root.root

    @cached_property
    def cuda(self) -> torch.device:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @property
    def docs(self):
        if self.saved_acts is None:
            raise ValueError("No saved_acts loaded; use Evaluation.from_cache_name(...) first")
        return self.saved_acts.tokens

    @property
    def docstrs(self) -> "StrDocs":
        return StrDocs(self)

    @property
    def acts(self):
        if self.saved_acts is None:
            raise ValueError("No saved_acts loaded; use Evaluation.from_cache_name(...) first")
        return self.saved_acts.acts

    @property
    def nnsight_model(self):
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.model

    @property
    def subject_model(self):
        return getattr(self.nnsight_model, "_model", self.nnsight_model)

    @property
    def path(self) -> Path:
        if self.saved_acts is None:
            raise ValueError("No cached acts loaded; call from_cache_name(...) or store_acts(...)")
        if self.filter is None:
            return self.cache_cfg.path
        return self.filter.filtered_dir(self.cache_cfg.path)

    @cached_property
    def cached_call(self) -> Union[CachedCalls, "Evaluation"]:
        return CachedCalls(self)

    @cached_property
    def metadatas(self) -> Metadatas:
        if self.filter is not None:
            raise NotImplementedError(
                "Metadatas on a filtered Evaluation are not implemented (use _root_metadatas)."
            )
        return Metadatas(self.path, cached_config=self.cache_cfg)

    @property
    def _root_metadatas(self) -> Metadatas:
        return self.root.metadatas if self._root is not None else self.metadatas

    @cached_property
    def artifacts(self) -> Artifacts:
        return Artifacts(self.path, cached_config=self.cache_cfg)

    @cached_property
    def filters(self) -> Filters:
        if self.filter is not None:
            raise ValueError("Cannot access filters from a filtered evaluation")
        return Filters(self.path, cached_config=self.cache_cfg)

    @property
    def d_dict(self) -> int:
        return self.architecture.run_cfg.init_cfg.d_dict

    @property
    def sae_cfg(self) -> RunConfig:
        return self.architecture.run_cfg

    @property
    def cache_cfg(self) -> CachingConfig:
        if self.saved_acts is None:
            raise ValueError("No saved_acts loaded")
        return self.saved_acts.cfg

    @property
    def features(self) -> "Features":
        if self.saved_acts is None or self.saved_acts.features is None:
            raise ValueError("No feature tensors available")
        return self.saved_acts.features

    @property
    def sae(self) -> "Trainable":
        return self.architecture.trainable

    @property
    def nnsight_site_name(self) -> str:
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.site

    @property
    def seq_len(self) -> int:
        return self.sae_cfg.train_cfg.data_cfg.seq_len

    @property
    def d_vocab(self) -> int:
        return self.tokenizer.vocab_size

    @takes_alias
    @classmethod
    def from_cache_name(cls, name: Path | str):
        if isinstance(name, str):
            name = Path(name)
            fp = Path.home() / "workspace" / "cached_sae_acts" / name
            if fp.exists():
                name = fp
            elif not name.exists():
                raise FileNotFoundError(f"Could not find cached acts at {name} or {fp}")

        saved = SavedActs[cls.get_inputs_type()].from_path(name)
        if saved.cfg.model_path is None:
            raise ValueError("cache_config.json missing model_path")
        inst = cls.from_model_path(path=saved.cfg.model_path, averaged_weights=saved.cfg.averaged_model_weights)
        inst.saved_acts = saved
        return inst

    @takes_alias
    @classmethod
    def from_model_path(cls, path: Path, averaged_weights: bool = False):
        path = path if isinstance(path, Path) else Path(path)
        arch = Architecture.load(path, load_weights=True, averaged_weights=averaged_weights)
        return cls(architecture=arch, model_path=path, averaged_model_weights=averaged_weights)

    def __attrs_post_init__(self):
        if self.saved_acts is not None and self.saved_acts.data_filter is not self.filter:
            raise ValueError("Filter mismatch between Evaluation and storage")
        if self.saved_acts is None and self.filter is not None:
            raise ValueError("Cannot set filter without saved_acts")

    def _apply_filter(self, filter_obj: NamedFilter | Tensor) -> "Evaluation":
        if isinstance(filter_obj, Tensor):
            filter_obj = NamedFilter(filter=filter_obj, filter_name=None)

        if self.filter is not None:
            raise ValueError("Filter already set; create filtered from the root Evaluation")

        return Evaluation(
            model_path=self.model_path,
            architecture=self.architecture,
            saved_acts=self.saved_acts.filtered(filter_obj),
            filter=filter_obj,
            root=self,
            tokenizer=self.tokenizer,
            model_adapter=self.model_adapter,
        )

    def open_filtered(self, filter_name: str) -> "Evaluation":
        return self._apply_filter(self.filters[filter_name])

    def store_acts(self, caching_cfg: CachingConfig, displace_existing: bool = False):
        if caching_cfg.model_path is None:
            caching_cfg.model_path = self.model_path
        if caching_cfg.model_path != self.model_path:
            raise ValueError("CachingConfig.model_path must match Evaluation.model_path")
        if caching_cfg.averaged_model_weights != self.averaged_model_weights:
            raise ValueError("Averaged weights mismatch")

        acts_cacher = ActsCacher.from_cache_and_runner(caching_config=caching_cfg, architecture=self.architecture)

        if acts_cacher.path.exists():
            if displace_existing:
                import time

                old = acts_cacher.path.parent / "old"
                old.mkdir(exist_ok=True, parents=True)
                stamp = f"{time.time():.6f}".replace(".", "_")
                acts_cacher.path.rename(old / f"{acts_cacher.path.name}_{stamp}")
            else:
                raise FileExistsError(f"{acts_cacher.path} already exists. Use displace_existing=True.")

        metadata_chunks = acts_cacher.store_acts()
        self.saved_acts = SavedActs[self.get_inputs_type()].from_path(acts_cacher.path)

        # Persist configured metadatas_from_src_column_names if present.
        metadata_builders = {
            name: self.metadata_builder(torch.long, "cpu")
            for name in self.cache_cfg.metadatas_from_src_column_names
        }
        for mchunk in metadata_chunks:
            for name, values in mchunk.items():
                metadata_builders[name].takestrl(values)

        for name, builder in metadata_builders.items():
            self.metadatas[name] = builder.value
            self.metadatas.set_str_translator(name, builder.unique_labels)

    def _metadata_for_doc_indices(
        self,
        doc_indices: Tensor | None,
        metadata: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        meta: dict[str, Tensor] = {} if metadata is None else dict(metadata)
        if doc_indices is None:
            return meta
        if not self.cache_cfg.metadatas_from_src_column_names:
            return meta

        idx = doc_indices.detach().cpu()
        for name in self.cache_cfg.metadatas_from_src_column_names:
            m = self._root_metadatas[name][idx]
            if isinstance(m, Tensor):
                meta[name] = m.to(self.cuda)
        return meta

    def _build_model_batch(self, tokens_or_batch, doc_indices: Tensor | None = None, metadata: dict[str, Tensor] | None = None):
        meta = self._metadata_for_doc_indices(doc_indices, metadata)

        def unwrap(x):
            if isinstance(x, FilteredTensor):
                return x.value
            if isinstance(x, DictBatch):
                return x.__class__.construct_with_other_data(
                    {k: unwrap(v) for k, v in x.items()},
                    x._get_other_dict(),
                )
            return x

        return self.model_adapter.make_batch(unwrap(tokens_or_batch), meta)

    def metadata_builder(
        self,
        dtype: torch.dtype,
        device: str | torch.device,
        item_size: list[int] | tuple[int, ...] = (),
    ) -> MetadataBuilder:
        return MetadataBuilder(
            self.saved_acts.chunks,  # type: ignore[union-attr]
            dtype=dtype,
            device=device,
            shape=[self.cache_cfg.num_docs, *item_size],
        )

    def filtered_builder(
        self,
        dtype: torch.dtype,
        device: str | torch.device,
        item_size: list[int] | None = None,
    ) -> FilteredBuilder:
        if self.filter is None:
            raise ValueError("filtered_builder can only be used on a filtered Evaluation")
        item_size = item_size or []
        return FilteredBuilder(
            self.saved_acts.chunks,  # type: ignore[union-attr]
            dtype=dtype,
            device=device,
            shape=[self.cache_cfg.num_docs, *item_size],
            filter=self.filter,
        )

    def get_features(self, feature_ids: list[int] | Tensor) -> list[FilteredTensor]:
        return [self.features[int(fid)] for fid in feature_ids]  # type: ignore[arg-type]

    # ---- labeling ----

    def get_feature_label(self, feature_id: int | str) -> str | None:
        return self.feature_labels.get(str(int(feature_id)))

    def set_feature_label(self, feature_id: int | str, label: str) -> None:
        self.feature_labels[str(int(feature_id))] = label

    def get_family_label(self, family: FamilyRef) -> str | None:
        return self.family_labels.get(str((int(family.level), int(family.family_id))))

    def set_family_label(self, family: FamilyRef, label: str) -> None:
        self.family_labels[str((int(family.level), int(family.family_id)))] = label

    def get_feature_model(self, feat_id: int | str) -> LabeledFeature:
        return LabeledFeature(feature_id=int(feat_id), label=self.get_feature_label(feat_id))

    # ---- token helpers ----

    def detokenize(self, tokens: int | list[int] | list[list[int]] | Tensor | DictBatch) -> list[str] | list[list[str]] | str:
        if isinstance(tokens, DictBatch):
            tokens = _extract_token_tensor(tokens)

        if isinstance(tokens, int):
            tokens = [tokens]
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        if not isinstance(tokens, Tensor):
            raise TypeError("detokenize expects int/list/Tensor/DictBatch")

        if tokens.ndim == 0:
            return self.tokenizer._tokenizer.decode([tokens.item()])
        if tokens.ndim == 1:
            return self.tokenizer._tokenizer.decode_batch([[int(t)] for t in tokens], skip_special_tokens=False)

        lens = tokens.shape[1]
        flat = einops.rearrange(tokens, "doc seq -> (doc seq)").unsqueeze(-1).tolist()
        flatl = self.tokenizer._tokenizer.decode_batch(flat, skip_special_tokens=False)
        return [flatl[i : i + lens] for i in range(0, len(flatl), lens)]

    # ---- dataset/feature stats ----

    def seq_aggregated_chunks_yielder(self, seq_agg: str) -> Generator[FilteredTensor]:
        for chunk in tqdm.tqdm(self.saved_acts.chunks):  # type: ignore[union-attr]
            acts = chunk.acts
            acts_inner = acts.value.to(self.cuda).to_dense()
            if acts_inner.ndim != 3:
                raise ValueError("Expected acts shaped (doc, seq, feat)")

            match seq_agg:
                case "count":
                    c_agg = (acts_inner > 0).sum(dim=1)
                case "any":
                    c_agg = (acts_inner > 0).any(dim=1)
                case "max":
                    c_agg = acts_inner.max(dim=1).values
                case "mean" | "sum":
                    c_agg = getattr(acts_inner, seq_agg)(dim=1)
                case _:
                    raise ValueError(f"Unknown seq_agg {seq_agg}")

            yield acts.to_filtered_like_self(c_agg)

    @property
    def num_docs(self) -> int:
        if self.filter:
            return int(self.filter.filter.sum().item())
        return self.cache_cfg.num_docs

    def acts_avg_over_dataset(self, seq_agg: str = "mean", docs_agg: str = "mean") -> Tensor:
        results = torch.zeros(self.d_dict, device=self.cuda)

        for agg_chunk in self.seq_aggregated_chunks_yielder(seq_agg):
            if docs_agg == "max":
                results = torch.cat([results, agg_chunk.value.max(dim=0).values]).max(dim=0).values
            elif docs_agg in ("mean", "sum"):
                results += agg_chunk.value.sum(dim=0)
            else:
                raise ValueError("docs_agg must be mean/max/sum")

        if docs_agg == "mean":
            results /= self.num_docs
        return results

    def get_feature(self, feature: int | FilteredTensor) -> Feature:
        if isinstance(feature, int):
            return Feature.make(src_eval=self, feature_id=feature)
        if isinstance(feature, FilteredTensor):
            return Feature.make(src_eval=self, feature=feature)
        raise TypeError(f"Invalid feature type: {type(feature)}")

    def chill_top_activations_and_metadatas(
        self,
        feature: int | FilteredTensor,
        p: float | None = None,
        k: int | None = None,
        return_acts_sparse: bool = False,
    ) -> TopActivations:
        f = self.get_feature(feature=feature)
        return f.top_activations(p=p, k=k)

    def _metadata_unique_labels_and_counts_tensor(self, key: str) -> MetadataLabelCounts:
        meta = self._root_metadatas[key]
        if self.filter is not None:
            meta = meta[self.filter.filter]
        if meta.ndim != 1 or meta.dtype != torch.long:
            raise ValueError("Expected 1D long metadata tensor")
        labels, counts = meta.unique(return_counts=True)
        return MetadataLabelCounts(key=key, labels=labels, counts=counts)

    def count_token_occurrence(self) -> Tensor:
        counts = torch.zeros(self.d_vocab, dtype=torch.long, device=self.cuda)
        for chunk in self.saved_acts.chunks:  # type: ignore[union-attr]
            toks = _extract_token_tensor(chunk.tokens.value).to(self.cuda).flatten()
            counts.scatter_add_(0, toks, torch.ones_like(toks, dtype=torch.long))
        return counts

    def num_active_docs_for_feature(self, feature_id: int) -> int:
        return int(self.cached_call._feature_num_active_docs()[feature_id].item())

    @torch.inference_mode()
    def _feature_num_active_tokens(self) -> Tensor:
        activity = torch.zeros(self.d_dict, dtype=torch.long, device=self.cuda)
        for chunk in self.saved_acts.chunks:  # type: ignore[union-attr]
            acts = chunk.acts.value.to(self.cuda).to_dense()
            activity += (acts > 0).sum(dim=1).sum(dim=0)
        return activity

    @property
    def seq_activation_counts(self) -> Tensor:
        return self.cached_call._feature_num_active_tokens().cpu()

    @property
    def seq_activation_probs(self) -> Tensor:
        return self.seq_activation_counts / (self.num_docs * self.seq_len)

    def _feature_num_active_docs(self) -> Tensor:
        activity = torch.zeros(self.d_dict, dtype=torch.long, device=self.cuda)
        for chunk in self.saved_acts.chunks:  # type: ignore[union-attr]
            acts = chunk.acts.value.to(self.cuda).to_dense()
            activity += (acts > 0).any(dim=1).sum(dim=0)
        return activity

    @property
    def doc_activation_counts(self) -> Tensor:
        return self.cached_call._feature_num_active_docs().cpu()

    def _feature_activity_sum(self) -> Tensor:
        activity = torch.zeros(self.d_dict, dtype=torch.float, device=self.cuda)
        for chunk in self.saved_acts.chunks:  # type: ignore[union-attr]
            acts = chunk.acts.value.to(self.cuda).to_dense()
            activity += acts.sum(dim=1).sum(dim=0)
        return activity

    @property
    def doc_activation_probs(self) -> Tensor:
        return self.doc_activation_counts / self.num_docs

    def get_metadata_intersection_filter_key(self, values: dict[str, str | list[str] | int | list[int]]) -> str:
        val_list = sorted(values.items(), key=lambda kv: kv[0])
        normalized: dict[str, tuple[int, ...]] = {}
        for k, v in val_list:
            if isinstance(v, (int, str)):
                v = [v]
            if isinstance(v, list) and v and isinstance(v[0], str):
                meta = self.metadatas.get(k)
                v = [meta.info.fromstr[x] for x in v]  # type: ignore[union-attr]
            normalized[k] = tuple(sorted(int(x) for x in v))
        key = str(normalized)

        if key not in self.filters:
            self.filters[key] = self._get_metadata_intersection_filter(normalized)
        return key

    def _get_metadata_intersection_filter(self, mapping: dict[str, tuple[int, ...]]) -> Tensor:
        filt = torch.ones(self.cache_cfg.num_docs, dtype=torch.bool, device=self.cuda)
        for mdname, values in mapping.items():
            md = self.metadatas[mdname].to(self.cuda)
            mdmask = torch.zeros_like(filt)
            for v in values:
                mdmask |= (md == v)
            filt &= mdmask
        return filt

    def top_activations_and_metadatas(
        self,
        feature: int | FilteredTensor,
        p: float | None = None,
        k: int | None = None,
        metadata_keys: list[str] | None = None,
        return_str_docs: bool = False,
        return_acts_sparse: bool = False,
        return_doc_indices: bool = True,
        str_metadatas: bool = False,
    ):
        metadata_keys = metadata_keys or []
        top_acts = self.chill_top_activations_and_metadatas(feature=feature, p=p, k=k, return_acts_sparse=return_acts_sparse)
        return self._legacy_top_activations_and_metadatas_getter(
            top_acts=top_acts,
            metadata_keys=metadata_keys,
            return_str_docs=return_str_docs,
            return_doc_indices=return_doc_indices,
            str_metadatas=str_metadatas,
        )

    def _legacy_top_activations_and_metadatas_getter(
        self,
        top_acts: TopActivations,
        metadata_keys: list[str],
        return_str_docs: bool,
        return_doc_indices: bool,
        str_metadatas: bool,
    ):
        acts = top_acts.acts
        docs = top_acts.doc_selection.doc_strs if return_str_docs else top_acts.docs
        metas = top_acts.doc_selection.metadata[metadata_keys]
        metadatas = metas.str_metadatas if str_metadatas else metas.metadatas

        if return_doc_indices:
            return docs, acts, metadatas, top_acts.doc_selection.doc_indices
        return docs, acts, metadatas


@define(slots=True)
class StrDocs:
    eval: Evaluation

    def __getitem__(self, idx: int | Tensor):
        toks = self.eval.docs[idx.cpu()]
        # toks may be Tensor or DictBatch. detokenize handles both.
        return self.eval.detokenize(toks)