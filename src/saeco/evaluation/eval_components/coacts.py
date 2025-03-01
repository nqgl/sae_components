from typing import TYPE_CHECKING

import einops
import torch
import tqdm
from saeco.evaluation.cache_version import cache_version
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
from torch import Tensor

if TYPE_CHECKING:
    from ..evaluation import Evaluation


class Coactivity:
    def activation_cosims(self: "Evaluation"):
        mat = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        f2sum = torch.zeros(self.d_dict).to(self.cuda)
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
            assert acts.ndim == 3
            feats_mat = einops.rearrange(acts, "doc seq feat -> feat (doc seq)")
            f2s = feats_mat.pow(2).sum(-1)
            assert f2s.shape == (self.d_dict,)
            f2sum = f2s + f2sum
            mat += feats_mat @ feats_mat.transpose(-2, -1)
        norms = f2sum.sqrt()
        mat /= norms.unsqueeze(0)
        mat /= norms.unsqueeze(1)
        prod = mat.diag()[~mat.diag().isnan()].prod()
        assert prod < 1.001 and prod > 0.999
        return mat

    def masked_activation_cosims(self: "Evaluation"):
        """
        Returns the masked cosine similarities matrix.
        Indexes are like: [masking feature, masked feature]
        """
        threshold = 0
        mat = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        f2sum = torch.zeros(self.d_dict).to(self.cuda)
        maskedf2sum = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
            feats_mat = einops.rearrange(acts, "doc seq feat -> feat (doc seq)")
            feats_mask = feats_mat > threshold

            f2s = feats_mat.pow(2).sum(-1)
            assert f2s.shape == (self.d_dict,)
            f2sum += f2s
            maskedf2sum += feats_mask.float() @ feats_mat.transpose(-2, -1).pow(2)
            mat += feats_mat @ feats_mat.transpose(-2, -1)
        norms = f2sum.sqrt()
        mat /= maskedf2sum.sqrt()
        mat /= norms.unsqueeze(1)
        prod = mat.diag()[~mat.diag().isnan()].prod()
        assert prod < 1.001 and prod > 0.999
        return mat

    def coactivations(self: "Evaluation", doc_agg=None):
        sims = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        coact_counts = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        fa_sq_sum = torch.zeros(self.d_dict).to(self.cuda)
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
            assert acts.ndim == 3
            feature_activity = self.sequelize(
                acts, doc_agg=doc_agg
            )  # feat, (doc [seq])
            feature_bin = (feature_activity > 0).float()
            fa_sq_sum += feature_activity.pow(2).sum(-1)
            sims += feature_activity @ feature_activity.transpose(-2, -1)
            coact_counts += feature_bin @ feature_bin.transpose(-2, -1)
        norms = fa_sq_sum.sqrt()
        sims /= norms.unsqueeze(0)
        sims /= norms.unsqueeze(1)
        prod = sims.diag()[~sims.diag().isnan()].prod()
        assert prod < 1.001 and prod > 0.999
        return coact_counts, sims

    def top_coactivating_features(self: "Evaluation", feature_id, top_n=10, mode="seq"):
        """
        mode: "seq" or "doc"
        """
        if mode == "seq":
            mat = self.cached_call.activation_cosims()
        elif mode == "doc":
            mat = self.cached_call.doc_level_co_occurrence()
        else:
            raise ValueError("mode must be 'seq' or 'doc'")
        vals = mat[feature_id]
        vals[feature_id] = -torch.inf
        vals[vals.isnan()] = -torch.inf
        top = vals.topk(top_n + 1)

        v = top.values
        i = top.indices
        v = v[i != feature_id][:top_n]
        i = i[i != feature_id][:top_n]
        return i, v

    def doc_level_co_occurrence(self: "Evaluation", pooling="mean"):
        """
        Pooling: "mean", "max" or "binary"
        this could be done at sequence level if we want
        """
        threshold = 0
        mat = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        f2sum = torch.zeros(self.d_dict).to(self.cuda)
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
            assert acts.ndim == 3
            if pooling == "mean":
                acts_pooled = acts.mean(1)
            elif pooling == "max":
                acts_pooled = acts.max(1).values
            elif pooling == "binary":
                acts_pooled = (acts > threshold).sum(1).float()
            feats_mat = einops.rearrange(acts_pooled, "doc feat -> feat doc")
            f2s = feats_mat.pow(2).sum(-1)
            assert f2s.shape == (self.d_dict,)
            f2sum = f2s + f2sum
            mat += feats_mat @ feats_mat.transpose(-2, -1)
        norms = f2sum.sqrt()
        mat /= norms.unsqueeze(0)
        mat /= norms.unsqueeze(1)
        prod = mat.diag()[~mat.diag().isnan()].prod()
        assert prod < 1.001 and prod > 0.999
        return mat

    def sequelize(
        self: "Evaluation",
        acts: Tensor,
        doc_agg: float | int | str | None = None,
    ):
        assert acts.ndim == 3
        if doc_agg:
            if isinstance(doc_agg, float | int):
                acts = acts.pow(doc_agg).sum(dim=1).pow(1 / doc_agg)
            elif doc_agg == "count":
                acts = (acts > 0).sum(dim=1).float()
            elif doc_agg == "max":
                acts = acts.max(dim=1).values
            else:
                raise ValueError("Invalid doc_agg")
            return einops.rearrange(acts, "doc feat -> feat doc")
        else:
            return einops.rearrange(acts, "doc seq feat -> feat (doc seq)")

    def cosims(self: "Evaluation", doc_agg=None):
        return self.coactivations(doc_agg=doc_agg)[1]

    def coactivity(self: "Evaluation", doc_agg=None):
        res = self.coactivations(doc_agg=doc_agg)
        self.artifacts[f"cosims({(doc_agg,)}, {{}})"] = res[1]
        return res[0]
