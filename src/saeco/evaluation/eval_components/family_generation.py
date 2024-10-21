from typing import TYPE_CHECKING

import einops
import torch
import tqdm

from saeco.evaluation.cached_artifacts import cache_version
from saeco.evaluation.fastapi_models.families_draft import (
    Family,
    FamilyLevel,
    FamilyRef,
    GetFamiliesResponse,
    ScoredFamilyRef,
    ScoredFeature,
)
from saeco.evaluation.fastapi_models.Feature import Feature

if TYPE_CHECKING:
    from ..evaluation import Evaluation


class FamilyGenerator:

    @torch.no_grad()
    def _get_feature_family_treesz(
        self, doc_agg=None, threshold=None, n=3, use_D=False, freq_bounds=None
    ):
        return torch.stack(
            self.generate_feature_families2(
                doc_agg=doc_agg,
                threshold=threshold,
                n=n,
                use_D=use_D,
                freq_bounds=freq_bounds,
            )[0]
        )

    @cache_version(10)
    def _get_feature_families_unlabeled(self) -> GetFamiliesResponse:
        from ..mst import Families, FamilyTreeNode

        # TODO .cached_call
        levels = self._get_feature_family_treesz()
        famlevels = [Families.from_tree(f) for f in levels]

        niceroots: list[list[FamilyTreeNode]] = [
            [r for r in f.roots if len(r) > 10]
            for i, f in enumerate(
                famlevels,
            )
        ]
        [
            len([r for r in f.roots if len(r) > 20 - i * 8])
            for i, f in enumerate(
                famlevels,
            )
        ]
        levels = []
        for levelnum, level in enumerate(niceroots):
            fl = FamilyLevel(
                level=levelnum,
                families={
                    fam_id: Family(
                        level=levelnum,
                        family_id=fam_id,
                        label=None,
                        subfamilies=[],
                        subfeatures=[
                            ScoredFeature(
                                feature=Feature(
                                    feature_id=int(feat_id),
                                    label=self.get_feature_label(feat_id),
                                ),
                                score=0.9,
                            )
                            for feat_id in root.family
                        ],
                    )
                    for fam_id, root in enumerate(level)
                },
            )
            levels.append(fl)
        level_lens = [len(l.families) for l in levels]
        # csll = torch.tensor([0] + level_lens).cumsum(0).tolist()[:-1]
        t0 = torch.zeros(
            len(levels),
            max(level_lens),
            self.d_dict,
            dtype=torch.float,
            device=self.cuda,
        )
        for i, level in enumerate(levels):
            for j, family in level.families.items():
                for feat in family.subfeatures:
                    feat: ScoredFeature
                    t0[i, j, feat.feature.feature_id] = 1
        ns = t0.sum(-1)

        t0 /= ns.unsqueeze(-1) + 1e-8
        sims = einops.einsum(t0, t0, "l1 f1 d, l2 f2 d -> l1 f1 l2 f2")
        # sims_f = einops.rearrange(sims, "l1 f1 l2 f2 -> l1 f1 (l2 f2)")
        # m = sims_f.max(dim=-1)

        # sml = sims.max(dim=-1)
        # smf = sml.values.max(dim=-1)
        # fi = smf.indices
        # li = sml.indices[fi]

        threshold = 0
        for i, level in enumerate(levels[:-1]):
            next_level = i + 1
            nl_sims = sims[i, :, next_level, :]
            z = torch.zeros_like(nl_sims)
            nlmax = nl_sims.max(dim=0)
            z[nlmax.indices, torch.arange(nlmax.indices.shape[0])] = nlmax.values
            z[z < threshold] = 0
            for j, family in level.families.items():
                # sim = sims[i, j, next_level, :]
                # st = sim > threshold
                st = z[j]
                # if st.sum() < 3:
                #     print("very few at threshold", threshold)
                #     st = sim > threshold / 2

                for f in st.nonzero():
                    family.subfamilies.append(
                        ScoredFamilyRef(
                            family=FamilyRef(
                                level=int(next_level),
                                family_id=int(f.item()),
                            ),
                            score=st[f.item()],
                        )
                    )
        return GetFamiliesResponse(levels=levels)

    def get_feature_families(self):
        ffs = self._get_feature_families_unlabeled()
        for level in ffs.levels:
            for family in level.families.values():
                family.label = self.get_family_label(family)
        ###
        # feature_level = FamilyLevel(
        #     level=3,
        #     families={
        #         i: Family(
        #             level=3,
        #             family_id=i,
        #             label=None,
        #             subfamilies=[],
        #             subfeatures=[],
        #         )
        #         for i in range(self.d_dict)
        #         if i < 1000
        #     },
        # )
        # ffs.levels.append(feature_level)
        # for family in ffs.levels[2].families.values():
        #     family.subfamilies.extend(
        #         [
        #             ScoredFamilyRef(
        #                 family=FamilyRef(level=3, family_id=feat.feature.feature_id),
        #                 score=feat.score,
        #             )
        #             for feat in family.subfeatures
        #         ]
        #     )
        # ###
        return ffs

    def generate_feature_families1(
        self: "Evaluation",
        doc_agg=None,
        threshold=0.1,
        n=3,
        use_D=False,
        freq_bounds=None,
    ):
        # C_unnormalized, D = self.coactivations(doc_agg=doc_agg)
        if use_D:
            unnormalized = self.cached_call.cosims(doc_agg=doc_agg).cpu()
        else:
            unnormalized = self.cached_call.coactivity(doc_agg=doc_agg).cpu()
        # D = D.cpu()
        C = unnormalized / (
            (
                feat_counts := (
                    self.doc_activation_counts
                    if doc_agg
                    else self.seq_activation_counts
                )
            )
            .cpu()
            .unsqueeze(-1)
            + 1e-6
        )
        threshold = threshold or C[C > 0].median()

        C[C.isnan()] = 0
        C[C < threshold] = 0
        if freq_bounds is not None:
            fmin, fmax = freq_bounds
            feat_probs = (
                self.doc_activation_probs if doc_agg else self.seq_activation_probs
            )
            bound = (feat_probs >= fmin) & (feat_probs <= fmax)
            C[~bound] = 0
            C[:, ~bound] = 0
        import scipy.sparse as ssp

        from ..mst import Families, FamilyTreeNode, mst, my_mst

        levels = []
        feat_counts = feat_counts.to(self.cuda)
        for _ in tqdm.trange(n):
            tree = mst(C).transpose(0, 1)
            roots = ((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0)
            # for i in range(nz.shape[0]):
            #     c = nz[i]
            #     assert feat_counts[c[0]] >= feat_counts[c[1]]
            # families = Families.from_tree(tree)
            levels.append(tree)
            C[roots] = 0
            C[:, roots] = 0
        # roots = [((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0) for tree in levels]

        return levels

    def generate_feature_families2(
        self: "Evaluation",
        doc_agg=None,
        threshold=0.1,
        n=3,
        use_D=False,
        freq_bounds=None,
        min_family_sizes=[20, 12, 7],
    ):
        # C_unnormalized, D = self.coactivations(doc_agg=doc_agg)
        if use_D:
            unnormalized = self.cached_call.cosims(doc_agg=doc_agg).cpu()
        else:
            unnormalized = self.cached_call.coactivity(doc_agg=doc_agg).cpu()
        # D = D.cpu()
        C = unnormalized / (
            (
                feat_counts := (
                    self.doc_activation_counts
                    if doc_agg
                    else self.seq_activation_counts
                )
            )
            .cpu()
            .unsqueeze(-1)
            + 1e-6
        )
        threshold = threshold or C[C > 0].median()

        C[C.isnan()] = 0
        C[C < threshold] = 0
        if freq_bounds is not None:
            fmin, fmax = freq_bounds
            feat_probs = (
                self.doc_activation_probs if doc_agg else self.seq_activation_probs
            )
            bound = (feat_probs >= fmin) & (feat_probs <= fmax)
            C[~bound] = 0
            C[:, ~bound] = 0
        import scipy.sparse as ssp

        from ..mst import Families, FamilyTreeNode, mst, my_mst

        levels = []
        families = []
        for i in range(n):
            tree = mst(C).transpose(0, 1)
            roots = ((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0)
            family = Families.from_tree(tree)
            kept = [r for r in family.roots if len(r) > min_family_sizes[i]]
            mask = torch.zeros_like(C, dtype=torch.bool)
            for f in tqdm.tqdm(kept):
                m = torch.zeros_like(mask[0])
                m[list(f.family)] = 1
                mask |= m.unsqueeze(0) & m.unsqueeze(1)
            C[~mask] = 0
            tree[~mask] = 0

            # for i in range(nz.shape[0]):
            #     c = nz[i]
            #     assert feat_counts[c[0]] >= feat_counts[c[1]]
            # families = Families.from_tree(tree)
            levels.append(tree)
            families.append(kept)
            C[roots] = 0
            C[:, roots] = 0
        # roots = [((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0) for tree in levels]

        return levels, families

    @torch.no_grad()
    def generate_feature_families(
        self: "Evaluation", doc_agg=None, threshold=None, n=3, use_D=False
    ):
        unnormalized = self.cached_call.coactivity(doc_agg=doc_agg).cpu()
        if use_D:
            unnormalized = self.cached_call.cosims(doc_agg=doc_agg).cpu()
        unnormalized[unnormalized.isnan()] = 0
        feat_counts = (
            self.doc_activation_counts if doc_agg else self.seq_activation_counts
        )
        # feat_probs = self.doc_activation_probs if doc_agg else self.seq_activation_probs

        def denan(x):
            return torch.where(x.isnan() | x.isinf(), torch.zeros_like(x), x)

        def zdiag(x):
            return torch.where(
                torch.eye(x.shape[0], device=x.device, dtype=torch.bool), 0, x
            )

        def isprob(P):
            assert (P >= 0).all() and (P <= 1).all()
            return P

        def probmat(P):
            return isprob(zdiag(denan(P)))

        def ent(P):
            P = isprob(denan(P))
            return torch.where(
                (P > 0) & (P < 1),
                -P * torch.log(P + 1e-6) - (1 - P) * torch.log(1 - P + 1e-6),
                0,
            )

        def nicemat(M):
            return zdiag(denan(M))

        def nent(Q, R):
            Q, R = nicemat(Q), nicemat(R)
            P = nicemat(Q / R)
            isprob(P)
            return torch.where(
                (P > 0) & (P < 1),
                -Q * torch.log(P) - (R - Q) * torch.log(1 - P),
                0,
            )

        N = self.num_docs if doc_agg else self.seq_len * self.num_docs
        A = feat_counts.unsqueeze(1).expand(-1, feat_counts.shape[0])
        B = feat_counts.unsqueeze(0).expand(feat_counts.shape[0], -1)
        V = unnormalized

        P_A = probmat(A / N)
        P_B = probmat(B / N)
        # P_AB = probmat(V / N)
        P_B_given_A = probmat((V + P_B * P_A) / (A + 1))  # +
        P_B_given_not_A = probmat((B - V + (1 - P_B) * (1 - P_A)) / (N - A + 1))
        # C = P_A * torch.log()
        # info = A * ent(P_B_given_A) + (N - A) * ent(P_B_given_not_A)
        # info = P_A * nent(V, A) + (1 - P_A) * nent((B - V), (N - A)) - nent(B, N)
        # info = (c := ent(P_B)) - (
        #     (a := P_A * ent(P_B_given_A)) + (b := (1 - P_A) * ent(P_B_given_not_A))
        # )
        # info = torch.where((P_B_given_A > P_B), info + 1e-6, 0)
        # (info[P_B_given_A > P_B]).sum()
        # (P_B_given_A > P_B).sum() / (P_B_given_A < P_B).sum()
        # r = P_A * torch.log(P_B_given_A / P_B)
        r = torch.log(P_B_given_A / P_B)
        r = ent(P_B) - ent(P_B_given_A)

        t = (1 - P_A) * torch.log(P_B_given_not_A / P_B)
        # other = P_A * torch.log(P_B_given_A / P_B) + (1 - P_A) * torch.log(
        #     P_B_given_not_A / P_B
        # )

        r = denan(r)
        t = denan(t)
        # r[P_B_given_A > P_B].sum()
        # r[P_B_given_A < P_B].sum()
        # t[P_B_given_A > P_B].sum()
        # t[P_B_given_A < P_B].sum()
        info = r

        # a.max(dim=0, keepdim=True)
        # b.max(dim=0, keepdim=True)
        # c.max(dim=0, keepdim=True)
        # a.max(dim=1, keepdim=True)
        # b.max(dim=1, keepdim=True)
        # c.max(dim=1, keepdim=True)
        # P_B.max()
        # v = B.to(torch.float64) / N
        # v.max()

        info = torch.where((V > 0) & (info > 0), info, 0)

        info = zdiag(denan(info))
        assert (info >= 0).all()
        # learned =
        # C = info.clone()
        C = info
        threshold = threshold if threshold is not None else C[C > 0].median()
        # C = unnormalized / (().cpu().unsqueeze(-1) + 1e-6)

        C.max(dim=0, keepdim=True)
        C.max(dim=1, keepdim=True)

        # C[C.isnan()] = 0
        C[C < threshold] = 0
        import scipy.sparse as ssp

        from ..mst import Families, FamilyTreeNode, mst, my_mst, prim_max

        levels = []
        feat_counts = feat_counts.to(self.cuda)
        for _ in range(n):
            # tree = mst(C)
            i, v = my_mst(C.cuda())
            tree = (
                torch.sparse_coo_tensor(indices=i, values=v, size=C.shape)
                .to_dense()
                .cpu()
                .transpose(0, 1)
            )
            nz = tree.nonzero()
            print(
                "proportion:",
                (feat_counts[nz[:, 0]] >= feat_counts[nz[:, 1]]).float().mean(),
            )

            # families = Families.from_tree(tree)
            roots = ((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0)
            print("num_roots", roots.sum())
            levels.append(tree)
            C[roots] = 0
            C[:, roots] = 0
        # roots.sum()

        # (
        #     feat_counts[levels[0].roots[0].feature_id],
        #     feat_counts[levels[1].roots[0].feature_id],
        #     feat_counts[levels[2].roots[0].feature_id],
        # )
        # (len(levels[0].roots[0]), len(levels[1].roots[0]), len(levels[2].roots[0]))

        # (len(levels[0]), len(levels[1]), len(levels[2]))

        return levels
