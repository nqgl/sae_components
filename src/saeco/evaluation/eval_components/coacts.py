from collections.abc import Callable
from typing import TYPE_CHECKING

import einops
import torch
import tqdm
from torch import Tensor

if TYPE_CHECKING:
    from ..evaluation import Evaluation


class Coactivity:
    # @torch.compile(dynamic=True)
    def coacts(
        self: "Evaluation",
        S: Callable[[Tensor], tuple[Tensor, Tensor]],
        reduce_prod: Callable[[Tensor, Tensor], Tensor],
        out_device: torch.device | str | None = None,
        f_chunk_i: int | None = None,
        f_chunk_j: int | None = None,
    ):
        if out_device is None:
            out_device = self.cuda
        if f_chunk_i is None:
            f_chunk_i = self.d_dict
        if f_chunk_j is None:
            f_chunk_j = self.d_dict

        mat = torch.zeros(self.d_dict, self.d_dict).to(out_device)
        f2sum = torch.zeros(self.d_dict).to(out_device)
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
            assert acts.ndim == 3
            # einops.rearrange(acts, "doc seq feat -> feat seq doc")
            s0, s1 = S(acts)
            # feat_indexed_S0 = S0.transpose(0, -1)  # doc seq feat -> feat seq doc
            # feat_indexed_S1 = S1.transpose(0, -1)  # doc seq feat -> feat seq doc
            for i in range(0, self.d_dict, f_chunk_i):
                act0s = s0.transpose(0, -1)[i : i + f_chunk_i].transpose(
                    0, -1
                )  # feat seq doc -> doc seq feat
                for j in range(0, self.d_dict, f_chunk_j):
                    acts1s = s1.transpose(0, -1)[j : j + f_chunk_j].transpose(
                        0, -1
                    )  # feat seq doc -> doc seq feat
                    res = reduce_prod(act0s, acts1s)
                    assert res.shape == (
                        min(f_chunk_i, self.d_dict - i),
                        min(f_chunk_j, self.d_dict - j),
                    )
                    mat[i : i + f_chunk_i, j : j + f_chunk_j] += res
                    f2sum += ...  # func that does this too prob

        return mat

    @torch.inference_mode()
    def activation_cosims(
        self: "Evaluation",
        out_device: torch.device | str | None = None,
        blocks_per_dim: int = 1,
    ):
        block_size = self.d_dict // blocks_per_dim
        if out_device is None:
            out_device = self.cuda
        mat = torch.zeros(self.d_dict, self.d_dict).to(out_device)
        f2sum = torch.zeros(self.d_dict).to(out_device)
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
            assert acts.ndim == 3
            feats_mat = einops.rearrange(acts, "doc seq feat -> feat (doc seq)")
            for i in range(0, self.d_dict, block_size):
                f2s = feats_mat[i : i + block_size].pow(2).sum(-1)
                # assert f2s.shape == (self.d_dict,)
                f2sum[i : i + block_size] += f2s.to(out_device)
                for j in range(blocks_per_dim):
                    mat[i : i + block_size, j : j + block_size] += (
                        feats_mat[i : i + block_size]
                        @ feats_mat[j : j + block_size].transpose(-2, -1)
                    ).to(out_device)
        norms = f2sum.sqrt()
        mat /= norms.unsqueeze(0)
        mat /= norms.unsqueeze(1)
        prod = mat.diag()[~mat.diag().isnan()].prod()
        print("prod", prod)
        # assert prod < 1.001 and prod > 0.999
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

    def coactivations(self: "Evaluation", doc_agg: float | int | str | None = None):
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
        assert prod < 1.001
        assert prod > 0.999
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
