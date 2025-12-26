from typing import TYPE_CHECKING

import torch
from torch import Tensor

from saeco.evaluation.fastapi_models.EnrichmentSortBy import (
    EnrichmentSortBy,
)
from saeco.evaluation.fastapi_models.metadata_enrichment import (
    MetadataEnrichmentLabelResult,
    MetadataEnrichmentResponse,
)
from saeco.evaluation.fastapi_models.token_enrichment import (
    TokenEnrichmentMode,
)
from saeco.evaluation.filtered import FilteredTensor

if TYPE_CHECKING:
    from ..evaluation import Evaluation


def score_enrichment(
    total_counts: torch.Tensor,
    total_denom: int | float,
    counts: torch.Tensor,
    sel_denom: torch.Tensor | float | int,
    smoothing=1,
    r=0.5,
    base_smoothing=1,
    r_base=0.5,
):
    assert total_counts.ndim == counts.ndim == 1
    base_rate = total_counts.float().mean() / total_denom
    p_total = (total_counts + base_smoothing * base_rate**r_base) / (
        total_denom + base_smoothing * base_rate ** (r_base - 1)
    )
    p_subset = (counts + smoothing * p_total**r) / (
        sel_denom + smoothing * p_total ** (r - 1)
    )
    return torch.log2(p_subset / p_total)


class Enrichment:
    @property
    def token_occurrence_count(self: "Evaluation") -> Tensor:
        return self.cached_call.count_token_occurrence()

    def top_activations_metadata_enrichments(
        self: "Evaluation",
        *,
        feature: int | FilteredTensor,
        metadata_keys: list[str],
        p: float = None,
        k: int = None,
        str_label: bool = False,
        sort_by: EnrichmentSortBy = EnrichmentSortBy.counts,
    ):
        top_acts = self.chill_top_activations_and_metadatas(
            feature=feature,
            p=p,
            k=k,
        )
        enrichments = top_acts.top_activations_metadata_enrichments(
            metadata_keys=metadata_keys,
        )
        results = {}
        for mdname, enrichment in enrichments.items():
            enrichment = enrichment.remove_zero_counts().sort(sort_by=sort_by)
            labels = (
                self._root_metadatas.get(mdname).strlist(enrichment.labels)
                if str_label
                else enrichment.labels.tolist()
            )
            results[mdname] = [
                MetadataEnrichmentLabelResult(
                    label=label,
                    count=count,
                    proportion=proportion,
                    normalized_count=normalized_count,
                    score=score,
                    # **(dict(act_sum=acts[md == label].sum()) if return_act_sum else {}),
                )
                for label, count, proportion, normalized_count, score in zip(
                    labels,
                    enrichment.counts.tolist(),
                    enrichment.proportions.tolist(),
                    enrichment.normalized_counts.tolist(),
                    enrichment.scores.tolist(),
                )
            ]
        return MetadataEnrichmentResponse(results=results)

    def top_activations_token_enrichments(
        self: "Evaluation",
        *,
        feature: int | FilteredTensor,
        p: float = None,
        k: int = None,
        mode: TokenEnrichmentMode = "doc",
        sort_by: EnrichmentSortBy = "count",
    ):
        docs, acts, metadatas, doc_ids = self.top_activations_and_metadatas(
            feature=feature, p=p, k=k, metadata_keys=[]
        )
        docs = docs.to(self.cuda)
        if mode == TokenEnrichmentMode.doc:
            seltoks = docs
        elif mode == TokenEnrichmentMode.max:
            max_pos = acts.argmax(dim=1)
            max_top = docs[torch.arange(max_pos.shape[0]), max_pos]
            seltoks = max_top
        elif mode == TokenEnrichmentMode.active:
            active_top = docs[acts > 0]
            seltoks = active_top
        elif mode == TokenEnrichmentMode.top:
            top_threshold = docs[
                acts > acts.max(dim=-1).values.min(dim=0).values.item()
            ]
            seltoks = top_threshold
        else:
            raise ValueError(f"Unknown mode {mode}")
        tokens, counts = seltoks.flatten().unique(return_counts=True, sorted=True)
        normalized_counts = (counts / seltoks.numel()) / (
            self.token_occurrence_count.to(self.cuda)[tokens]
            / (self.num_docs * self.seq_len)
        )
        scores = score_enrichment(
            counts=counts,
            sel_denom=seltoks.numel(),
            total_counts=self.token_occurrence_count.to(self.cuda)[tokens],
            total_denom=self.num_docs * self.seq_len,
        )
        if sort_by == EnrichmentSortBy.counts:
            i = counts.argsort(descending=True)
        elif sort_by == EnrichmentSortBy.normalized_count:
            i = normalized_counts.argsort(descending=True)
        elif sort_by == EnrichmentSortBy.score:
            i = scores.argsort(descending=True)
        else:
            raise ValueError(f"Unknown sort_by {sort_by}")
        tokens = tokens[i]
        counts = counts[i]
        normalized_counts = normalized_counts[i]
        scores = scores[i]

        return tokens, counts, normalized_counts, scores
