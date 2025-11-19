from typing import TYPE_CHECKING

import einops
import torch
import tqdm

from saeco.evaluation.fastapi_models.families_draft import (
    Family,
    FamilyLevel,
    FamilyRef,
    GetFamiliesResponse,
    ScoredFamilyRef,
    ScoredFeature,
)
from saeco.evaluation.fastapi_models.Feature import Feature
from saeco.evaluation.fastapi_models.metadata_enrichment import (
    MetadataEnrichmentLabelResult,
    MetadataEnrichmentResponse,
    MetadataEnrichmentSortBy,
)
from saeco.evaluation.fastapi_models.token_enrichment import (
    TokenEnrichmentMode,
    TokenEnrichmentSortBy,
)
from saeco.evaluation.filtered import FilteredTensor

if TYPE_CHECKING:
    from ..evaluation import Evaluation


class Enrichment:
    def top_activations_metadata_enrichments(
        self: "Evaluation",
        *,
        feature: int | FilteredTensor,
        metadata_keys: list[str],
        p: float = None,
        k: int = None,
        str_label: bool = False,
        sort_by: MetadataEnrichmentSortBy = MetadataEnrichmentSortBy.counts,
    ):
        docs, acts, metadatas, doc_ids = self.top_activations_and_metadatas(
            feature=feature, p=p, k=k, metadata_keys=metadata_keys
        )
        r = {}
        num_docs = doc_ids.shape[0]
        for mdname, md in metadatas.items():
            assert md.ndim == 1
            full_lc = self.cached_call._metadata_unique_labels_and_counts_tensor(mdname)
            labels, mdcat_counts = torch.cat([md, full_lc[0]]).unique(
                return_counts=True  # adds one of each of all labels to it so that it has all metadatas and is consistently indexed with the full counts
            )
            counts = mdcat_counts - 1
            assert (labels == full_lc[0]).all()
            assert counts.shape == labels.shape == full_lc[1].shape
            proportions = counts / full_lc[1]
            scores = self.score_enrichment(
                counts=counts,
                sel_denom=num_docs,
                total_counts=full_lc[1],
                total_denom=self.num_docs,
            )
            scores = scores[counts > 0]
            labels = labels[counts > 0]
            proportions = proportions[counts > 0]
            counts = counts[counts > 0]
            normalized_counts = proportions * self.num_docs / doc_ids.shape[0]
            if sort_by == TokenEnrichmentSortBy.counts:
                i = counts.argsort(descending=True)
            elif sort_by == TokenEnrichmentSortBy.normalized_count:
                i = normalized_counts.argsort(descending=True)
            elif sort_by == TokenEnrichmentSortBy.score:
                i = scores.argsort(descending=True)
            else:
                raise ValueError(f"Unknown sort_by {sort_by}")
            labels = labels[i]
            counts = counts[i]
            proportions = proportions[i]
            normalized_counts = normalized_counts[i]
            scores = scores[i]
            r[mdname] = [
                MetadataEnrichmentLabelResult(
                    label=label,
                    count=count,
                    proportion=proportion,
                    normalized_count=normalized_count,
                    score=score,
                    # **(dict(act_sum=acts[md == label].sum()) if return_act_sum else {}),
                )
                for label, count, proportion, normalized_count, score in zip(
                    (
                        self.metadatas.get(mdname).strlist(labels)
                        if str_label
                        else labels.tolist()
                    ),
                    counts.tolist(),
                    proportions.tolist(),
                    normalized_counts.tolist(),
                    scores.tolist(),
                )
            ]
        return MetadataEnrichmentResponse(results=r)

    def top_activations_token_enrichments(
        self: "Evaluation",
        *,
        feature: int | FilteredTensor,
        p: float = None,
        k: int = None,
        mode: TokenEnrichmentMode = "doc",
        sort_by: TokenEnrichmentSortBy = "count",
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
        scores = self.score_enrichment(
            counts=counts,
            sel_denom=seltoks.numel(),
            total_counts=self.token_occurrence_count.to(self.cuda)[tokens],
            total_denom=self.num_docs * self.seq_len,
        )
        if sort_by == TokenEnrichmentSortBy.counts:
            i = counts.argsort(descending=True)
        elif sort_by == TokenEnrichmentSortBy.normalized_count:
            i = normalized_counts.argsort(descending=True)
        elif sort_by == TokenEnrichmentSortBy.score:
            i = scores.argsort(descending=True)
        else:
            raise ValueError(f"Unknown sort_by {sort_by}")
        tokens = tokens[i]
        counts = counts[i]
        normalized_counts = normalized_counts[i]
        scores = scores[i]

        return tokens, counts, normalized_counts, scores

    def score_enrichment(
        self: "Evaluation",
        total_counts: torch.Tensor,
        total_denom: int | float,
        counts: torch.Tensor,
        sel_denom: torch.Tensor,
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
