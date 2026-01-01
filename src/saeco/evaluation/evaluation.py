from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from attrs import Factory, define, field
from paramsight import get_resolved_typevars_for_base, takes_alias
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from saeco.architecture.architecture import Architecture
from saeco.data.dict_batch import DictBatch
from saeco.evaluation.eval_components.perturbation_analysis import PerturbationAnalysis
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
from .views import DecodedTextView, MetadataView, TokenStringsView

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

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        if self._tokenizer is not None:
            return self._tokenizer
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.tokenizer

    def _model_adapter_default(self):
        model_kwargs = getattr(
            self.sae_cfg.train_cfg.data_cfg.model_cfg, "model_kwargs", {}
        )
        try:
            from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig

            if isinstance(
                self.sae_cfg.train_cfg.data_cfg.model_cfg.model_load_cfg,
                ComlmModelConfig,
            ):  # type: ignore[attr-defined]
                return ComlmEvalAdapter(model_kwargs=model_kwargs)
        except Exception:
            pass
        return LanguageModelEvalAdapter(model_kwargs=model_kwargs)

    model_adapter: ModelEvalAdapter = field(
        default=Factory(_model_adapter_default, takes_self=True)
    )

    # -----------------------
    # Constructors
    # -----------------------

    @takes_alias
    @classmethod
    def get_inputs_type(cls) -> type[InputsT]:
        return get_resolved_typevars_for_base(cls, Evaluation)[0]  # type: ignore

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

        inst = cls.open_model(
            cached.cfg.model_path, averaged_weights=cached.cfg.averaged_model_weights
        )
        inst.cached_acts = cached
        return inst

    @classmethod
    def open_model(
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
    def cached(self) -> CachedCalls | Evaluation:
        return CachedCalls(self)

    # -----------------------
    # Data access
    # -----------------------

    @property
    def samples(self):
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

    # -----------------------
    # Views
    # -----------------------

    @property
    def text(self) -> DecodedTextView:
        return DecodedTextView(self)

    @property
    def token_strs(self) -> TokenStringsView:
        return TokenStringsView(self)

    @property
    def metadata(self) -> MetadataView:
        return MetadataView(self)

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
                "Use eval.metadata[...] on filtered evals (metadatas store is root-only)"
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

    def open_filter(self, name: str) -> Evaluation:
        return self.root._apply_filter(self.filter_store.get_filter(name))

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
        f: int | FilteredTensor,
        *,
        k: int | None = None,
        p: float | None = None,
        agg=None,
    ) -> TopActivations:
        feat = self.feature(f)
        if agg is None:
            return feat.top_activations(k=k, p=p)
        return feat.top_activations(agg=agg, k=k, p=p)

    # -----------------------
    # Counts / stats
    # -----------------------

    @property
    def num_samples(self) -> int:
        if self.filter is not None:
            return int(self.filter.filter.sum().item())
        return self.cache_config.num_docs

    def count_token_occurrence(self) -> Tensor:
        counts = torch.zeros(self.d_vocab, dtype=torch.long, device=self.device)
        for chunk in self.cached_acts.chunks:  # type: ignore[union-attr]
            toks = extract_token_tensor(chunk.tokens.value).to(self.device).flatten()
            counts.scatter_add_(0, toks, torch.ones_like(toks, dtype=torch.long))
        return counts

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
