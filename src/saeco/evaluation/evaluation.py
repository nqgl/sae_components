from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from attrs import Factory, define, field
from paramsight import get_resolved_typevars_for_base, takes_alias
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from saeco.architecture.sae_architecture import Architecture
from saeco.data.dict_batch import DictBatch
from saeco.evaluation.eval_components.perturbation_analysis import (
    PerturbationAnalysis,
    PerturbationConfig,
)
from saeco.evaluation.model_interface import (
    ComlmEvalAdapter,
    LanguageModelEvalAdapter,
    ModelEvalAdapter,
)
from saeco.trainer import RunConfig

from .BMStorShelf import BMStorShelf
from .cached_artifacts import CachedCalls
from .eval_components.coacts import Coactivity
from .eval_components.enrichment import Enrichment
from .eval_components.patching import Patching
from .filtered import FilteredTensor
from .named_filter import NamedFilter
from .return_objects import Feature, TopActivations
from .storage.cache_config import CacheConfig
from .storage.cached_acts import CachedActs
from .storage.cacher import ActsCacher
from .storage.stored_metadata import Artifacts, Filters, Metadatas
from .token_utils import extract_token_tensor
from .views import MetadataView, TokenStringsView

if TYPE_CHECKING:
    from saeco.evaluation.features import Features


@define(slots=True)
class Evaluation[InputsT: torch.Tensor | DictBatch](
    Enrichment, Patching, Coactivity, PerturbationAnalysis
):
    model_path: Path
    architecture: Architecture = field(repr=False)
    averaged_model_weights: bool = field(default=False, repr=False)

    cached_acts: CachedActs | None = field(default=None, repr=False)
    filter: NamedFilter | None = field(default=None)
    _root: Evaluation | None = field(default=None, repr=False, alias="root")
    _tokenizer: PreTrainedTokenizerFast | None = field(default=None, repr=False)
    _perturbation_config: PerturbationConfig | None = field(default=None, repr=False)

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        if self._tokenizer is not None:
            return self._tokenizer
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.tokenizer

    def _model_adapter_default(self):
        try:
            from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig

            if isinstance(
                self.sae_cfg.train_cfg.data_cfg.model_cfg.model_load_cfg,
                ComlmModelConfig,
            ):  # type: ignore[attr-defined]
                return ComlmEvalAdapter(eval=self)
        except Exception:
            pass
        return LanguageModelEvalAdapter(eval=self)

    model_adapter: ModelEvalAdapter = field(
        default=Factory(_model_adapter_default, takes_self=True)
    )

    def __attrs_post_init__(self):
        if (
            self.cached_acts is not None
            and self.cached_acts.data_filter is not self.filter
        ):
            raise ValueError("Filter mismatch between Evaluation and storage")
        if self.cached_acts is None and self.filter is not None:
            raise ValueError("Cannot set filter without cached_acts")

    # -----------------------
    # Constructors
    # -----------------------

    @takes_alias
    @classmethod
    def get_inputs_type(cls) -> type[InputsT]:
        return get_resolved_typevars_for_base(cls, Evaluation)[0]  # type: ignore

    @takes_alias
    @classmethod
    def open_cache(cls, name: Path | str) -> Evaluation:
        if isinstance(name, str):
            name = Path(name)
            fp = Path.home() / "workspace" / "cached_sae_acts" / name
            if fp.exists():
                name = fp
            elif not name.exists():
                raise FileNotFoundError(f"Could not find cached acts at {name} or {fp}")

        cached = CachedActs[cls.get_inputs_type()].open(name)
        if cached.cfg.model_path is None:
            raise ValueError("cache_config.json missing model_path")

        inst = cls.open_from_model(
            cached.cfg.model_path, averaged_weights=cached.cfg.averaged_model_weights
        )
        inst.cached_acts = cached
        return inst

    @takes_alias
    @classmethod
    def open_from_model(
        cls, path: Path | str, *, averaged_weights: bool = False
    ) -> Evaluation:
        p = path if isinstance(path, Path) else Path(path)
        arch = Architecture.load(
            p, load_weights=True, averaged_weights=averaged_weights
        )
        return cls(
            architecture=arch, model_path=p, averaged_model_weights=averaged_weights
        )

    # -----------------------
    # Core properties
    # -----------------------

    @property
    def root(self) -> Evaluation:
        return self if self._root is None else self._root.root

    @cached_property
    def device(self) -> torch.device:
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @property
    def sae_cfg(self) -> RunConfig:
        return self.architecture.run_cfg

    @property
    def cache_config(self) -> CacheConfig:
        if self.cached_acts is None:
            raise ValueError("No cached_acts loaded")
        return self.cached_acts.cfg

    @property
    def path(self) -> Path:
        if self.cached_acts is None:
            raise ValueError(
                "No cached_acts loaded; call Evaluation.open_cache(...) first"
            )
        if self.filter is None:
            return self.cache_config.path
        return self.filter.filtered_dir(self.cache_config.path)

    @cached_property
    def bmstore(self) -> BMStorShelf:
        return BMStorShelf.from_path(self.path)

    @cached_property
    def feature_labels(self):
        import shelve

        p = self.root.path
        p.mkdir(parents=True, exist_ok=True)
        return shelve.open(str(p / "feature_labels"))

    @cached_property
    def family_labels(self):
        import shelve

        p = self.path
        p.mkdir(parents=True, exist_ok=True)
        return shelve.open(str(p / "family_labels"))

    @cached_property
    def cached(self) -> CachedCalls | Evaluation:
        return CachedCalls(self)

    # -----------------------
    # Data access
    # -----------------------

    @property
    def docs(self):
        if self.cached_acts is None:
            raise ValueError("No cached_acts loaded")
        return self.cached_acts.tokens

    @property
    def acts(self):
        if self.cached_acts is None:
            raise ValueError("No cached_acts loaded")
        return self.cached_acts.acts

    @property
    def nnsight_model(self):
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.model

    @property
    def subject_model(self):
        return getattr(self.nnsight_model, "_model", self.nnsight_model)

    @property
    def nnsight_site_name(self) -> str:
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.site

    @property
    def seq_len(self) -> int:
        seq_len = self.sae_cfg.train_cfg.data_cfg.seq_len
        assert seq_len is not None
        return seq_len

    @property
    def d_dict(self) -> int:
        return self.architecture.run_cfg.init_cfg.d_dict

    @property
    def d_vocab(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def sae(self):
        """Returns the trainable SAE model."""
        return self.architecture.trainable

    # -----------------------
    # Views
    # -----------------------

    @property
    def doc_strs(self) -> TokenStringsView:
        return TokenStringsView(eval=self)

    @property
    def metadata(self) -> MetadataView:
        return MetadataView(eval=self)

    # -----------------------
    # Stores (root-scoped)
    # -----------------------

    @cached_property
    def artifact_store(self) -> Artifacts:
        return Artifacts(self.path, cache_config=self.cache_config)

    @cached_property
    def metadata_store(self) -> Metadatas:
        if self.filter is not None:
            raise ValueError(
                "Use eval.metadata[...] on filtered evals (metadata_store is root-only)"
            )
        return Metadatas(self.path, cache_config=self.cache_config)

    @property
    def _root_metadatas(self) -> Metadatas:
        return (
            self.root.metadata_store if self._root is not None else self.metadata_store
        )

    @cached_property
    def filter_store(self) -> Filters:
        # Always root store
        return Filters(self.root.path, cache_config=self.cache_config)

    # -----------------------
    # Filtering
    # -----------------------

    def save_filter(self, name: str, mask: Tensor) -> NamedFilter:
        """Persist a boolean doc-mask in the root filter store under `name`."""
        mask = mask.detach().to(dtype=torch.bool, device="cpu")
        self.filter_store[name] = mask
        return self.filter_store[name]

    def open_filter(self, name: str) -> Evaluation:
        """Open a named filter from the root store and return a filtered Evaluation."""
        return self.root._apply_filter(self.filter_store.get_filter(name))

    def where(self, mask: Tensor) -> Evaluation:
        """Create an ephemeral filtered evaluation with an unnamed filter."""
        nf = NamedFilter(
            filter=mask.detach().to(dtype=torch.bool, device="cpu"), filter_name=None
        )
        return self.root._apply_filter(nf)

    def _apply_filter(self, filter_obj: NamedFilter | Tensor) -> Evaluation:
        if isinstance(filter_obj, Tensor):
            filter_obj = NamedFilter(filter=filter_obj, filter_name=None)
        if self.filter is not None:
            raise ValueError("Filter already set; filter from root instead")
        if self.cached_acts is None:
            raise ValueError("No cached_acts loaded")

        return Evaluation(
            model_path=self.model_path,
            architecture=self.architecture,
            cached_acts=self.cached_acts.filtered(filter_obj),
            filter=filter_obj,
            root=self,
            tokenizer=self.tokenizer,
            model_adapter=self.model_adapter,
        )

    # -----------------------
    # Token decoding
    # -----------------------

    def decode_text(
        self,
        tokens: int | list[int] | Tensor | DictBatch,
        *,
        skip_special_tokens: bool = False,
    ) -> str | list[str]:
        if isinstance(tokens, int):
            return self.tokenizer.decode(
                [tokens], skip_special_tokens=skip_special_tokens
            )

        if isinstance(tokens, list):
            t = torch.tensor(tokens, dtype=torch.long)
        else:
            t = extract_token_tensor(tokens)

        if t.ndim == 0:
            return self.tokenizer.decode(
                [int(t.item())], skip_special_tokens=skip_special_tokens
            )
        if t.ndim == 1:
            return self.tokenizer.decode(
                t.tolist(), skip_special_tokens=skip_special_tokens
            )
        if t.ndim == 2:
            return self.tokenizer.batch_decode(
                t.tolist(), skip_special_tokens=skip_special_tokens
            )
        raise ValueError(f"decode_text expects ndim 0/1/2, got {t.ndim}")

    def detokenize(
        self, tokens: int | list[int] | Tensor | DictBatch
    ) -> str | list[str] | list[list[str]]:
        if isinstance(tokens, int):
            return self.tokenizer.convert_ids_to_tokens(tokens)

        if isinstance(tokens, list):
            t = torch.tensor(tokens, dtype=torch.long)
        else:
            t = extract_token_tensor(tokens)

        if t.ndim == 0:
            return self.tokenizer.convert_ids_to_tokens(int(t.item()))
        if t.ndim == 1:
            return self.tokenizer.convert_ids_to_tokens(t.tolist())
        if t.ndim == 2:
            return [self.tokenizer.convert_ids_to_tokens(row.tolist()) for row in t]
        raise ValueError(f"token_strings expects ndim 0/1/2, got {t.ndim}")

    # -----------------------
    # Features
    # -----------------------

    @property
    def features(self) -> Features:
        if self.cached_acts is None or self.cached_acts.features is None:
            raise ValueError("No feature tensors available")
        return self.cached_acts.features

    def feature(self, f: int | FilteredTensor) -> Feature:
        if isinstance(f, int):
            return Feature.make(src_eval=self, feature_id=f)
        if isinstance(f, FilteredTensor):
            return Feature.make(src_eval=self, feature=f)
        raise TypeError(f"Invalid feature spec: {type(f)}")

    def top_activations(
        self,
        feature: int | FilteredTensor,
        *,
        k: int | None = None,
        p: float | None = None,
        agg=None,
    ) -> TopActivations:
        feat = self.feature(feature)
        if agg is None:
            return feat.top_activations(k=k, p=p)
        return feat.top_activations(agg=agg, k=k, p=p)

    def get_features(self, feature_ids: list[int] | Tensor) -> list[FilteredTensor]:
        return [self.features[int(fid)] for fid in feature_ids]

    # -----------------------
    # Labeling
    # -----------------------

    def get_feature_label(self, feature_id: int | str) -> str | None:
        return self.feature_labels.get(str(int(feature_id)))

    def set_feature_label(self, feature_id: int | str, label: str) -> None:
        self.feature_labels[str(int(feature_id))] = label

    def get_feature_model(self, feat_id: int | str):
        from .fastapi_models import LabeledFeature

        return LabeledFeature(
            feature_id=int(feat_id), label=self.get_feature_label(feat_id)
        )

    # -----------------------
    # Counts / stats
    # -----------------------

    @property
    def num_docs(self) -> int:
        if self.filter is not None:
            return int(self.filter.filter.sum().item())
        return self.cache_config.num_docs

    def count_token_occurrence(self) -> Tensor:
        counts = torch.zeros(self.d_vocab, dtype=torch.long, device=self.device)
        for chunk in self.cached_acts.chunks:  # type: ignore[union-attr]
            toks = extract_token_tensor(chunk.tokens.value).to(self.device).flatten()
            counts.scatter_add_(0, toks, torch.ones_like(toks, dtype=torch.long))
        return counts

    def seq_aggregated_chunks_yielder(self, seq_agg: str):
        import tqdm as tqdm_module

        for chunk in tqdm_module.tqdm(self.cached_acts.chunks):  # type: ignore[union-attr]
            acts = chunk.acts
            acts_inner = acts.value.to(self.device).to_dense()
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

    def acts_avg_over_dataset(
        self, seq_agg: str = "mean", docs_agg: str = "mean"
    ) -> Tensor:
        results = torch.zeros(self.d_dict, device=self.device)

        for agg_chunk in self.seq_aggregated_chunks_yielder(seq_agg):
            if docs_agg == "max":
                results = (
                    torch.cat([results, agg_chunk.value.max(dim=0).values])
                    .max(dim=0)
                    .values
                )
            elif docs_agg in ("mean", "sum"):
                results += agg_chunk.value.sum(dim=0)
            else:
                raise ValueError("docs_agg must be mean/max/sum")

        if docs_agg == "mean":
            results /= self.num_docs
        return results

    @torch.inference_mode()
    def _feature_num_active_tokens(self) -> Tensor:
        activity = torch.zeros(self.d_dict, dtype=torch.long, device=self.device)
        for chunk in self.cached_acts.chunks:  # type: ignore[union-attr]
            acts = chunk.acts.value.to(self.device).to_dense()
            activity += (acts > 0).sum(dim=1).sum(dim=0)
        return activity

    @property
    def seq_activation_counts(self) -> Tensor:
        return self.cached._feature_num_active_tokens().cpu()

    @property
    def seq_activation_probs(self) -> Tensor:
        return self.seq_activation_counts / (self.num_docs * self.seq_len)

    def _feature_num_active_docs(self) -> Tensor:
        activity = torch.zeros(self.d_dict, dtype=torch.long, device=self.device)
        for chunk in self.cached_acts.chunks:  # type: ignore[union-attr]
            acts = chunk.acts.value.to(self.device).to_dense()
            activity += (acts > 0).any(dim=1).sum(dim=0)
        return activity

    @property
    def doc_activation_counts(self) -> Tensor:
        return self.cached._feature_num_active_docs().cpu()

    def _feature_activity_sum(self) -> Tensor:
        activity = torch.zeros(self.d_dict, dtype=torch.float, device=self.device)
        for chunk in self.cached_acts.chunks:  # type: ignore[union-attr]
            acts = chunk.acts.value.to(self.device).to_dense()
            activity += acts.sum(dim=1).sum(dim=0)
        return activity

    @property
    def doc_activation_probs(self) -> Tensor:
        return self.doc_activation_counts / self.num_docs

    def num_active_docs_for_feature(self, feature_id: int) -> int:
        return int(self.cached._feature_num_active_docs()[feature_id].item())

    # -----------------------
    # Storing acts (unchanged behavior, renamed config type)
    # -----------------------

    def store_acts(self, cfg: CacheConfig, displace_existing: bool = False):
        if cfg.model_path is None:
            cfg.model_path = self.model_path
        if cfg.model_path != self.model_path:
            raise ValueError("CacheConfig.model_path must match Evaluation.model_path")
        if cfg.averaged_model_weights != self.averaged_model_weights:
            raise ValueError("Averaged weights mismatch")

        cacher = ActsCacher.from_cache_and_runner(
            caching_config=cfg, architecture=self.architecture
        )

        if cacher.path.exists():
            if not displace_existing:
                raise FileExistsError(
                    f"{cacher.path} already exists. Use displace_existing=True."
                )
            import time

            old = cacher.path.parent / "old"
            old.mkdir(exist_ok=True, parents=True)
            stamp = f"{time.time():.6f}".replace(".", "_")
            cacher.path.rename(old / f"{cacher.path.name}_{stamp}")

        cacher.store_acts()
        self.cached_acts = CachedActs[self.get_inputs_type()].open(cacher.path)

    # -----------------------
    # Metadata utilities
    # -----------------------

    def metadata_tensor(
        self, key: str, *, device: torch.device | str | None = None
    ) -> Tensor:
        """Get metadata aligned to this Evaluation's doc space (filtered-aware)."""
        md = self._root_metadatas[key]
        if self.filter is not None:
            mask = self.filter.filter.detach().cpu()
            md = md[mask]
        if device is not None:
            md = md.to(device)
        return md

    def metadata_builder(
        self,
        dtype: torch.dtype,
        device: str | torch.device,
        item_size: list[int] | tuple[int, ...] = (),
    ):
        from .storage.MetadataBuilder import MetadataBuilder

        return MetadataBuilder(
            self.cached_acts.chunks,  # type: ignore[union-attr]
            dtype=dtype,
            device=device,
            shape=[self.cache_config.num_docs, *item_size],
        )

    def filtered_builder(
        self,
        dtype: torch.dtype,
        device: str | torch.device,
        item_size: list[int] | None = None,
    ):
        from .storage.MetadataBuilder import FilteredBuilder

        if self.filter is None:
            raise ValueError(
                "filtered_builder can only be used on a filtered Evaluation"
            )
        item_size = item_size or []
        return FilteredBuilder(
            self.cached_acts.chunks,  # type: ignore[union-attr]
            dtype=dtype,
            device=device,
            shape=[self.cache_config.num_docs, *item_size],
            filter=self.filter,
        )

    def _metadata_for_doc_indices(
        self,
        doc_indices: Tensor | None,
        metadata: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        meta: dict[str, Tensor] = {} if metadata is None else dict(metadata)
        if doc_indices is None:
            return meta
        if not self.cache_config.metadatas_from_src_column_names:
            return meta

        idx = doc_indices.detach().cpu()
        for name in self.cache_config.metadatas_from_src_column_names:
            m = self._root_metadatas[name][idx]
            if isinstance(m, Tensor):
                meta[name] = m.to(self.device)
        return meta

    def _build_model_batch(
        self,
        tokens_or_batch,
        doc_indices: Tensor | None = None,
        metadata: dict[str, Tensor] | None = None,
    ):
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

    def _metadata_unique_labels_and_counts_tensor(self, key: str):
        from .return_objects import MetadataLabelCounts

        meta = self._root_metadatas[key]
        if self.filter is not None:
            meta = meta[self.filter.filter]
        if meta.ndim != 1 or meta.dtype != torch.long:
            raise ValueError("Expected 1D long metadata tensor")
        labels, counts = meta.unique(return_counts=True)
        return MetadataLabelCounts(key=key, labels=labels, counts=counts)

    def get_metadata_intersection_filter_key(
        self, values: dict[str, str | list[str] | int | list[int]]
    ) -> str:
        val_list = sorted(values.items(), key=lambda kv: kv[0])
        normalized: dict[str, tuple[int, ...]] = {}
        for k, v in val_list:
            if isinstance(v, (int, str)):
                v = [v]
            if isinstance(v, list) and v and isinstance(v[0], str):
                meta = self.metadata_store.get(k)
                v = [meta.info.fromstr[x] for x in v]  # type: ignore[union-attr]
            normalized[k] = tuple(sorted(int(x) for x in v))
        key = str(normalized)

        if key not in self.filter_store:
            self.filter_store[key] = self._get_metadata_intersection_filter(normalized)
        return key

    def _get_metadata_intersection_filter(
        self, mapping: dict[str, tuple[int, ...]]
    ) -> Tensor:
        filt = torch.ones(
            self.cache_config.num_docs, dtype=torch.bool, device=self.device
        )
        for mdname, values in mapping.items():
            md = self.metadata_store[mdname].to(self.device)
            mdmask = torch.zeros_like(filt)
            for v in values:
                mdmask |= md == v
            filt &= mdmask
        return filt

    # -----------------------
    # Legacy
    # -----------------------
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
        if metadata_keys is None:
            metadata_keys = []
        top_acts = self.top_activations(
            feature=feature,
            p=p,
            k=k,
        )
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
        metadatas = top_acts.doc_selection.metadata[list(metadata_keys)]
        if str_metadatas:
            metadatas = metadatas.str_metadatas
        else:
            metadatas = metadatas.metadatas

        if return_doc_indices:
            doc_indices = top_acts.doc_selection.doc_indices
            return docs, acts, metadatas, doc_indices
        return docs, acts, metadatas
