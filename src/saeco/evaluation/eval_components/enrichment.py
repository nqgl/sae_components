from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from saeco.evaluation.fastapi_models.EnrichmentSortBy import EnrichmentSortBy
from saeco.evaluation.fastapi_models.metadata_enrichment import (
    MetadataEnrichmentLabelResult,
    MetadataEnrichmentResponse,
)
from saeco.evaluation.fastapi_models.token_enrichment import TokenEnrichmentMode
from saeco.evaluation.filtered import FilteredTensor
from saeco.evaluation.token_utils import extract_token_tensor

if TYPE_CHECKING:
    from saeco.evaluation.evaluation import Evaluation


def score_enrichment(
    total_counts: Tensor,
    total_denom: int | float,
    counts: Tensor,
    sel_denom: Tensor | float | int,
    smoothing: float = 10,
    r: float = 0.5,
    base_smoothing: float = 10,
    r_base: float = 0.5,
) -> Tensor:
    if total_counts.ndim != 1 or counts.ndim != 1:
        raise ValueError("score_enrichment expects 1D total_counts and counts")

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
    def token_occurrence_count(self: Evaluation) -> Tensor:
        return self.cached.count_token_occurrence()

    def top_activations_metadata_enrichments(
        self: Evaluation,
        *,
        feature: int | FilteredTensor,
        metadata_keys: list[str],
        p: float | None = None,
        k: int | None = None,
        str_label: bool = False,
        sort_by: EnrichmentSortBy = EnrichmentSortBy.counts,
    ) -> MetadataEnrichmentResponse:
        top_acts = self.top_activations(feature=feature, p=p, k=k)
        enrichments = top_acts.top_activations_metadata_enrichments(
            metadata_keys=metadata_keys
        )

        results: dict[str, list[MetadataEnrichmentLabelResult]] = {}
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
                )
                for label, count, proportion, normalized_count, score in zip(
                    labels,
                    enrichment.counts.tolist(),
                    enrichment.proportions.tolist(),
                    enrichment.normalized_counts.tolist(),
                    enrichment.scores.tolist(),
                    strict=True,
                )
            ]

        return MetadataEnrichmentResponse(results=results)

    def top_activations_token_enrichments(
        self: Evaluation,
        *,
        feature: int | FilteredTensor,
        p: float | None = None,
        k: int | None = None,
        mode: TokenEnrichmentMode = TokenEnrichmentMode.doc,
        sort_by: EnrichmentSortBy = EnrichmentSortBy.counts,
    ):
        top_acts = self.top_activations(feature=feature, p=p, k=k)
        tokens = top_acts.docs
        acts = top_acts.acts
        tokens = extract_token_tensor(tokens).to(self.device)

        if mode == TokenEnrichmentMode.doc:
            seltoks = tokens
        elif mode == TokenEnrichmentMode.max:
            max_pos = acts.argmax(dim=1)
            seltoks = tokens[torch.arange(max_pos.shape[0]), max_pos]
        elif mode == TokenEnrichmentMode.active:
            seltoks = tokens[acts.indices().unbind()]
        elif mode == TokenEnrichmentMode.top:
            threshold = acts.max(dim=-1).values.min(dim=0).values.item()
            seltoks = tokens[acts.to_dense() > threshold]
        else:
            raise ValueError(f"Unknown mode {mode}")

        tokens, counts = seltoks.flatten().unique(return_counts=True, sorted=True)

        normalized_counts = (counts / seltoks.numel()) / (
            self.token_occurrence_count.to(self.device)[tokens]
            / (self.num_docs * self.seq_len)
        )

        scores = score_enrichment(
            counts=counts,
            sel_denom=seltoks.numel(),
            total_counts=self.token_occurrence_count.to(self.device)[tokens],
            total_denom=self.num_docs * self.seq_len,
        )

        match sort_by:
            case EnrichmentSortBy.counts:
                order = counts.argsort(descending=True)
            case EnrichmentSortBy.normalized_count:
                order = normalized_counts.argsort(descending=True)
            case EnrichmentSortBy.score:
                order = scores.argsort(descending=True)
            case _:
                raise ValueError(f"Unknown sort_by {sort_by}")

        return tokens[order], counts[order], normalized_counts[order], scores[order]
