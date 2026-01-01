from __future__ import annotations

from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Self

import torch
from attrs import define, field
from torch import Tensor

from saeco.data.dict_batch import DictBatch
from saeco.evaluation.eval_components.enrichment import score_enrichment
from saeco.evaluation.fastapi_models.EnrichmentSortBy import EnrichmentSortBy

if TYPE_CHECKING:
    from saeco.evaluation.evaluation import Evaluation
    from saeco.evaluation.filtered import FilteredTensor
    from saeco.evaluation.fastapi_models.families_draft import FamilyRef


def _pk_to_k(p: float | None, k: int | None, quantity: int) -> int:
    if (p is None) == (k is None):
        raise ValueError("Exactly one of p and k must be set")
    if p is not None and not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    if k is None:
        k = int(quantity * p)
    if k <= 0:
        raise ValueError("k must be positive")
    return min(k, quantity)


@define(slots=True)
class EvalRefData:
    src_eval: "Evaluation"

    @property
    def device(self) -> torch.device:
        return self.src_eval.cuda


@define(slots=True)
class FeatureSpec:
    feature_id: int

    def open(self, src_eval: "Evaluation") -> "FilteredTensor":
        return src_eval.features[self.feature_id]


class AggregationType(Enum):
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    ANY = "any"


@define(slots=True)
class Feature(EvalRefData):
    spec: FeatureSpec | "FilteredTensor"

    @cached_property
    def data(self) -> "FilteredTensor":
        if isinstance(self.spec, FeatureSpec):
            return self.spec.open(self.src_eval)
        return self.spec

    @classmethod
    def make(
        cls,
        src_eval: "Evaluation",
        feature_id: int | None = None,
        feature: "FilteredTensor | None" = None,
    ) -> Self:
        if (feature_id is None) == (feature is None):
            raise ValueError("Exactly one of feature_id and feature must be set")
        return cls(src_eval=src_eval, spec=feature if feature is not None else FeatureSpec(feature_id=feature_id))  # type: ignore[arg-type]

    def aggregate(self, agg: AggregationType) -> "FilteredTensor":
        def _dense(x: Tensor) -> Tensor:
            return x.to_dense() if x.is_sparse else x

        match agg:
            case AggregationType.MEAN:
                return self.data.apply_to_inner(lambda x: _dense(x).mean(dim=1), cut_to_ndim=1)
            case AggregationType.MAX:
                return self.data.apply_to_inner(lambda x: _dense(x).max(dim=1).values, cut_to_ndim=1)
            case AggregationType.SUM:
                return self.data.apply_to_inner(lambda x: _dense(x).sum(dim=1), cut_to_ndim=1)
            case AggregationType.COUNT:
                return self.data.apply_to_inner(lambda x: (_dense(x) > 0).sum(dim=1), cut_to_ndim=1)
            case AggregationType.ANY:
                return self.data.apply_to_inner(lambda x: (_dense(x) > 0).any(dim=1).to(dtype=torch.float32), cut_to_ndim=1)
            case _:
                raise ValueError(f"Invalid aggregation type: {agg}")

    def top(self, *args, **kwargs) -> "TopActivations":
        return self.top_activations(*args, **kwargs)

    def top_activations(
        self,
        agg: AggregationType = AggregationType.MAX,
        *,
        p: float | None = None,
        k: int | None = None,
    ) -> "TopActivations":
        doc_acts = self.aggregate(agg)
        k = _pk_to_k(p, k, int(doc_acts.value.shape[0]))  # type: ignore[attr-defined]

        vals = doc_acts.value  # type: ignore[attr-defined]
        topk = vals.topk(k, sorted=True)

        top_outer_indices = doc_acts.externalize_indices(topk.indices.unsqueeze(0))
        doc_indices = top_outer_indices[0]

        return TopActivations(
            src_eval=self.src_eval,
            feature=self,
            doc_selection=SelectedDocs(doc_indices=doc_indices, src_eval=self.src_eval),
        )


@define(slots=True)
class MetadataAccessor(EvalRefData):
    doc_selection: "SelectedDocs"
    metadatas: dict[str, "SelectedMetadata"] = field(factory=dict)

    def __getitem__(self, key: str | list[str]) -> "SelectedMetadata | SelectedMetadatas":
        if isinstance(key, list):
            return SelectedMetadatas(
                selected_metadatas={k: self[k] for k in key}, src_eval=self.src_eval
            )
        if key not in self.metadatas:
            self.metadatas[key] = SelectedMetadata(
                src_eval=self.src_eval,
                docs=self.doc_selection,
                key=key,
            )
        return self.metadatas[key]


@define(slots=True)
class SelectedMetadata(EvalRefData):
    docs: "SelectedDocs"
    key: str

    @cached_property
    def data(self) -> Tensor:
        return self.src_eval._root_metadatas[self.key][self.docs.doc_indices.cpu()]

    @cached_property
    def as_str(self) -> Tensor | list[str]:
        return self.src_eval._root_metadatas.translate({self.key: self.data})[self.key]


@define(slots=True)
class SelectedMetadatas(EvalRefData):
    selected_metadatas: dict[str, SelectedMetadata]

    @cached_property
    def metadatas(self) -> dict[str, Tensor]:
        return {k: v.data for k, v in self.selected_metadatas.items()}

    @cached_property
    def str_metadatas(self) -> dict[str, Tensor | list[str]]:
        return {k: v.as_str for k, v in self.selected_metadatas.items()}


@define(slots=True)
class SelectedDocs:
    doc_indices: Tensor
    src_eval: "Evaluation"

    @cached_property
    def metadata(self) -> MetadataAccessor:
        return MetadataAccessor(doc_selection=self, src_eval=self.src_eval)

    @property
    def docs(self) -> Tensor | DictBatch:
        return self.src_eval.docs[self.doc_indices]

    @property
    def texts(self) -> str | list[str]:
        return self.src_eval.text[self.doc_indices]

    @property
    def token_strs(self):
        return self.src_eval.token_strs[self.doc_indices]

    @property
    def doc_strs(self):
        return self.src_eval.docstrs[self.doc_indices]


@DictBatch.auto_other_fields
class MetadataEnrichmentResult(DictBatch):
    name: str
    labels: torch.Tensor
    counts: torch.Tensor
    proportions: torch.Tensor
    normalized_counts: torch.Tensor
    scores: torch.Tensor

    def remove_zero_counts(self):
        return self[self.counts > 0]

    def sort(self, sort_by: EnrichmentSortBy = EnrichmentSortBy.counts):
        match sort_by:
            case EnrichmentSortBy.counts:
                i = self.counts.argsort(descending=True)
            case EnrichmentSortBy.normalized_count:
                i = self.normalized_counts.argsort(descending=True)
            case EnrichmentSortBy.score:
                i = self.scores.argsort(descending=True)
            case _:
                raise ValueError(f"Unknown sort_by {sort_by}")
        return self[i]


@define(slots=True)
class TopActivations(EvalRefData):
    feature: Feature
    doc_selection: SelectedDocs

    @classmethod
    def make(cls, src_eval: "Evaluation", feature: "FilteredTensor", doc_indices: Tensor) -> Self:
        return cls(
            src_eval=src_eval,
            feature=Feature(src_eval=src_eval, spec=feature),
            doc_selection=SelectedDocs(src_eval=src_eval, doc_indices=doc_indices),
        )

    @cached_property
    def acts(self) -> Tensor | DictBatch:
        return self.feature.data.index_select(self.doc_selection.doc_indices, dim=0)

    @property
    def docs(self) -> Tensor | DictBatch:
        return self.doc_selection.docs

    @property
    def texts(self):
        return self.doc_selection.texts

    @property
    def token_strs(self):
        return self.doc_selection.token_strs

    @property
    def doc_strs(self):
        return self.doc_selection.doc_strs

    def top_activations_metadata_enrichments(
        self,
        *,
        metadata_keys: list[str],
    ) -> dict[str, MetadataEnrichmentResult]:
        metadatas = self.doc_selection.metadata[metadata_keys].metadatas
        doc_ids = self.doc_selection.doc_indices
        num_docs = int(doc_ids.shape[0])

        out: dict[str, MetadataEnrichmentResult] = {}
        for mdname, md in metadatas.items():
            if md.ndim != 1:
                raise ValueError("Metadata enrichment expects 1D metadata")

            metadata_counts = self.src_eval.cached_call._metadata_unique_labels_and_counts_tensor(mdname)
            labels, mdcat_counts = torch.cat([md, metadata_counts.labels]).unique(return_counts=True)

            counts = mdcat_counts - 1  # remove the one-of-each we added
            if not (labels == metadata_counts.labels).all():
                raise RuntimeError("Label alignment mismatch in enrichment calculation")

            proportions = counts / metadata_counts.counts
            scores = score_enrichment(
                counts=counts,
                sel_denom=num_docs,
                total_counts=metadata_counts.counts,
                total_denom=self.src_eval.num_docs,
            )
            out[mdname] = MetadataEnrichmentResult(
                name=mdname,
                labels=labels,
                counts=counts,
                proportions=proportions,
                normalized_counts=proportions * self.src_eval.num_docs / num_docs,
                scores=scores,
            )
        return out


@DictBatch.auto_other_fields
class MetadataLabelCounts(DictBatch):
    key: str
    labels: torch.Tensor
    counts: torch.Tensor