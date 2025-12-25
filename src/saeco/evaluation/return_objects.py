from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Self, overload

import torch
from attrs import define, field
from torch import Tensor

from saeco.data.dict_batch.dict_batch import DictBatch
from saeco.evaluation.eval_components.enrichment import score_enrichment
from saeco.evaluation.fastapi_models.EnrichmentSortBy import EnrichmentSortBy

if TYPE_CHECKING:
    from saeco.evaluation.evaluation import Evaluation
    from saeco.evaluation.filtered import FilteredTensor


def _pk_to_k(p: float | None, k: int | None, quantity: int) -> int:
    if (p is None) == (k is None):
        raise ValueError("Exactly one of p and k must be set")
    if p is not None and not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    if k is None:
        assert p is not None
        k = int(quantity * p)
    if k <= 0:
        raise ValueError("k must be positive")
    return min(k, quantity)


@define
class EvalRefData:
    src_eval: "Evaluation"

    @property
    def device(self) -> torch.device:
        return self.src_eval.cuda

    ### NOTE:
    # if memory management is an issue due to the cached properties,
    # add method to this to clear all cached properties
    # and also possibly register each into a weakrefdict on the base eval
    # so base eval can clear all the props if needed


@define
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


@define
class Feature(EvalRefData):
    spec: "FeatureSpec | FilteredTensor"

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
        if feature is not None:
            if feature_id is not None:
                raise ValueError("Exactly one of feat_id and feature must be set")
            return cls(src_eval=src_eval, spec=feature)
        if feature_id is None:
            raise ValueError("Exactly one of feat_id and feature must be set")

        return cls(src_eval=src_eval, spec=FeatureSpec(feature_id=feature_id))

    @classmethod
    def make_batch(
        cls,
        src_eval: "Evaluation",
        features: list[int | "FilteredTensor"],
    ) -> list[Self]:
        """
        Create multiple Feature objects efficiently.

        Args:
            src_eval: The Evaluation object
            features: List of feature IDs or FilteredTensor objects

        Returns:
            List of Feature objects
        """
        return [
            cls.make(
                src_eval=src_eval,
                feature_id=f if isinstance(f, int) else None,
                feature=f if isinstance(f, FilteredTensor) else None,
            )
            for f in features
        ]

    def aggregate(self, agg: AggregationType):
        match agg:
            case AggregationType.MEAN:

                def fn(x: Tensor):
                    return x.to_dense().mean(dim=1)
            case AggregationType.MAX:

                def fn(x: Tensor):
                    return x.to_dense().max(dim=1).values
            case AggregationType.SUM:

                def fn(x: Tensor):
                    return x.to_dense().sum(dim=1)
            case AggregationType.COUNT:

                def fn(x: Tensor):
                    return (x > 0).to_dense().sum(dim=1)
            case _:
                raise ValueError(f"Invalid aggregation type: {agg}")
        return self.data.apply_to_inner(fn, cut_to_ndim=1)

    def top_activations(
        self,
        agg: AggregationType = AggregationType.MAX,
        *,
        p: float | None = None,
        k: int | None = None,
    ) -> "TopActivations":
        doc_acts = self.aggregate(agg)
        k = _pk_to_k(p, k, doc_acts.value.shape[0])
        topk = doc_acts.value.topk(k, sorted=True)
        top_outer_indices = doc_acts.externalize_indices(topk.indices.unsqueeze(0))
        doc_indices = top_outer_indices[0]
        return TopActivations(
            src_eval=self.src_eval,
            feature=self,
            doc_selection=SelectedDocs(
                doc_indices=doc_indices,
                src_eval=self.src_eval,
            ),
        )

    @staticmethod
    def batched_top_activations(
        features: list["Feature"],
        agg: AggregationType = AggregationType.MAX,
        *,
        p: float | None = None,
        k: int | None = None,
    ) -> list["TopActivations"]:
        """
        Compute top activations for multiple features efficiently.

        This method processes multiple features together, enabling
        more efficient GPU utilization when fetching top activations
        for many features.

        Args:
            features: List of Feature objects to process
            agg: Aggregation type (MAX, MEAN, SUM, COUNT)
            p: Proportion of top activations (alternative to k)
            k: Number of top activations

        Returns:
            List of TopActivations objects
        """
        if not features:
            return []

        # All features should have the same src_eval
        src_eval = features[0].src_eval

        # Step 1: Aggregate all features (GPU-bound operations)
        doc_acts_list = [f.aggregate(agg) for f in features]

        # Step 2: Compute topk for each feature
        results = []
        for feature, doc_acts in zip(features, doc_acts_list):
            actual_k = _pk_to_k(p, k, doc_acts.value.shape[0])
            topk = doc_acts.value.topk(actual_k, sorted=True)
            top_outer_indices = doc_acts.externalize_indices(topk.indices.unsqueeze(0))
            doc_indices = top_outer_indices[0]

            top_act = TopActivations(
                src_eval=src_eval,
                feature=feature,
                doc_selection=SelectedDocs(
                    doc_indices=doc_indices,
                    src_eval=src_eval,
                ),
            )
            results.append(top_act)

        return results

    # @cached_property
    # def id(self) -> int:
    #     # oh maybe list[int] or something more flexible to do more complex citation?
    #     assert self.feature.slicing is not None
    #     i = self.feature.slicing.slices[2]
    #     assert isinstance(i, int)
    #     return i


@define
class DocSelectionCitation:
    feature_id: int
    topk: int


@define
class MetadataAccessor(EvalRefData):
    doc_selection: "SelectedDocs"
    metadatas: dict[str, "SelectedMetadata"] = field(factory=dict)

    @overload
    def __getitem__(self, key: str) -> "SelectedMetadata": ...
    @overload
    def __getitem__(self, key: list[str]) -> "SelectedMetadatas": ...
    def __getitem__(
        self, key: str | list[str]
    ) -> "SelectedMetadata| SelectedMetadatas":
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


@define
class SelectedMetadata(EvalRefData):
    docs: "SelectedDocs"
    key: str

    @cached_property
    def data(self):
        # TODO: check in on the return raw thing (eg below is caused by it)
        return self.src_eval._root_metadatas[self.key][self.docs.doc_indices.cpu()]

    @cached_property
    def as_str(self) -> Tensor | list[str]:
        return self.src_eval._root_metadatas.translate({self.key: self.data})[self.key]


@define
class SelectedMetadatas(EvalRefData):
    selected_metadatas: dict[str, "SelectedMetadata"]

    @cached_property
    def metadatas(self) -> dict[str, Tensor]:
        return {k: v.data for k, v in self.selected_metadatas.items()}

    @cached_property
    def str_metadatas(self) -> dict[str, Tensor | list[str]]:
        return {k: v.as_str for k, v in self.selected_metadatas.items()}


@define
class SelectedDocs:
    doc_indices: Tensor  # so if you had more data sources,
    # this could totally be a dictbatch
    # of dataset id -> indices for that dataset
    # or no maybe just a pair of (name, indices)

    src_eval: "Evaluation"

    @cached_property
    def metadata(self) -> MetadataAccessor:
        return MetadataAccessor(
            doc_selection=self,
            src_eval=self.src_eval,
        )

    @property
    def docs(self) -> Tensor | DictBatch:
        return self.src_eval.docs[self.doc_indices]

    @property
    def doc_strs(self) -> list[str] | str | list[list[str]]:
        return self.src_eval.docstrs[self.doc_indices]


from saeco.data import DictBatch


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


@define
class TopActivations(EvalRefData):
    feature: Feature
    doc_selection: SelectedDocs
    # info about k?

    @classmethod
    def make(
        cls,
        src_eval: "Evaluation",
        feature: "FilteredTensor",
        doc_indices: Tensor,
    ) -> Self:
        return cls(
            src_eval=src_eval,
            feature=Feature(
                src_eval=src_eval,
                spec=feature,
            ),
            doc_selection=SelectedDocs(
                src_eval=src_eval,
                doc_indices=doc_indices,
            ),
        )

    @cached_property
    def acts(self) -> Tensor | DictBatch:
        return self.feature.data.index_select(self.doc_selection.doc_indices, dim=0)

    @property
    def docs(self) -> Tensor | DictBatch:
        return self.doc_selection.docs

    @property
    def doc_strs(self) -> list[str] | str | list[list[str]]:
        return self.doc_selection.doc_strs

    def top_activations_metadata_enrichments(
        self,
        *,
        metadata_keys: list[str],
    ):
        # docs = self.docs
        # acts = self.acts
        metadatas = self.doc_selection.metadata[list(metadata_keys)].metadatas
        doc_ids = self.doc_selection.doc_indices
        r = {}
        num_docs = doc_ids.shape[0]
        for mdname, md in metadatas.items():
            assert md.ndim == 1
            metadata_counts = (
                self.src_eval.cached_call._metadata_unique_labels_and_counts_tensor(
                    mdname
                )
            )
            labels, mdcat_counts = torch.cat(
                [md, metadata_counts.labels]
                # adds one of each of all labels to it so
                # it has all metadatas and is consistently indexed with the full counts
            ).unique(return_counts=True)
            assert isinstance(labels, Tensor)
            assert isinstance(mdcat_counts, Tensor)
            counts = mdcat_counts - 1  # remove the 1 of each labels added
            assert (labels == metadata_counts.labels).all()
            assert counts.shape == labels.shape == metadata_counts.counts.shape

            proportions = counts / metadata_counts.counts
            scores = score_enrichment(
                counts=counts,
                sel_denom=num_docs,
                total_counts=metadata_counts.counts,
                total_denom=self.src_eval.num_docs,
            )
            r[mdname] = MetadataEnrichmentResult(
                name=mdname,
                labels=labels,
                counts=counts,
                proportions=proportions,
                normalized_counts=proportions
                * self.src_eval.num_docs
                / doc_ids.shape[0],
                scores=scores,
            )
        return r

    # def top_activations_token_enrichments(
    #     self: "Evaluation",
    #     *,
    #     feature: int | FilteredTensor,
    #     p: float = None,
    #     k: int = None,
    #     mode: TokenEnrichmentMode = "doc",
    #     sort_by: EnrichmentSortBy = "count",
    # ):
    #     docs, acts, metadatas, doc_ids = self.top_activations_and_metadatas(
    #         feature=feature, p=p, k=k, metadata_keys=[]
    #     )
    #     docs = docs.to(self.cuda)
    #     if mode == TokenEnrichmentMode.doc:
    #         seltoks = docs
    #     elif mode == TokenEnrichmentMode.max:
    #         max_pos = acts.argmax(dim=1)
    #         max_top = docs[torch.arange(max_pos.shape[0]), max_pos]
    #         seltoks = max_top
    #     elif mode == TokenEnrichmentMode.active:
    #         active_top = docs[acts > 0]
    #         seltoks = active_top
    #     elif mode == TokenEnrichmentMode.top:
    #         top_threshold = docs[
    #             acts > acts.max(dim=-1).values.min(dim=0).values.item()
    #         ]
    #         seltoks = top_threshold
    #     else:
    #         raise ValueError(f"Unknown mode {mode}")
    #     tokens, counts = seltoks.flatten().unique(return_counts=True, sorted=True)
    #     normalized_counts = (counts / seltoks.numel()) / (
    #         self.token_occurrence_count.to(self.cuda)[tokens]
    #         / (self.num_docs * self.seq_len)
    #     )
    #     scores = score_enrichment(
    #         counts=counts,
    #         sel_denom=seltoks.numel(),
    #         total_counts=self.token_occurrence_count.to(self.cuda)[tokens],
    #         total_denom=self.num_docs * self.seq_len,
    #     )
    #     if sort_by == EnrichmentSortBy.counts:
    #         i = counts.argsort(descending=True)
    #     elif sort_by == EnrichmentSortBy.normalized_count:
    #         i = normalized_counts.argsort(descending=True)
    #     elif sort_by == EnrichmentSortBy.score:
    #         i = scores.argsort(descending=True)
    #     else:
    #         raise ValueError(f"Unknown sort_by {sort_by}")
    #     tokens = tokens[i]
    #     counts = counts[i]
    #     normalized_counts = normalized_counts[i]
    #     scores = scores[i]

    #     return tokens, counts, normalized_counts, scores


@DictBatch.auto_other_fields
class MetadataLabelCounts(DictBatch):
    key: str
    labels: torch.Tensor
    counts: torch.Tensor
