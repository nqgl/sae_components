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
from saeco.evaluation.fastapi_models.Feature import LabeledFeature

if TYPE_CHECKING:
    from ..evaluation import Evaluation
from attrs import define
from torch import Tensor

INFOC_VERSION = 4
MAIN_VERSION = 26


@define
class Controller: ...


class Node:
    feature_id: int
    controller: Controller
    root_distances: Tensor


def distances(tree: Tensor, roots: Tensor):
    tree = tree + tree.transpose(0, 1)
    tree = torch.where(tree > 0, tree, torch.inf)

    root_ids = roots.nonzero()[:, 0]
    dists = torch.zeros(
        roots.sum(),
        tree.shape[0],
        device=tree.device,
    )
    dists[:] = torch.inf

    for i in tqdm.trange(root_ids.shape[0]):
        root = root_ids[i]
        dist = dists[i]
        dist[root] = 0
        visited = torch.zeros_like(roots, dtype=torch.bool)
        # print("root", root)
        while True:
            nextmask = (dist != torch.inf) & ~visited
            if not nextmask.any():
                break
            tree[nextmask].shape
            v = tree[nextmask]
            v[:, visited] = torch.inf
            m = v.min(dim=1)
            # print(m.indices)
            notinf = v != torch.inf
            z = torch.where(notinf, (v + dist[nextmask].unsqueeze(-1)), 0).sum(0)

            dist[notinf.any(0)] = z[notinf.any(0)]
            # print((dist != torch.inf).sum())
            visited |= nextmask
            if visited.all():
                break

    return dists


def connectedness(tree: Tensor, roots: Tensor):
    assert not (tree > 1).any()
    tree = tree + tree.transpose(0, 1)
    tree = torch.where(tree > 0, tree, 0)
    tree.diag()[:] = 0

    root_ids = roots.nonzero()[:, 0]
    dists = torch.zeros(
        roots.sum(),
        tree.shape[0],
        device=tree.device,
    )

    for i in tqdm.trange(root_ids.shape[0]):
        root = root_ids[i]
        dist = dists[i]
        dist[root] = 1
        visited = torch.zeros_like(roots, dtype=torch.bool)
        # print("root", root)
        while True:
            nextmask = (dist != 0) & ~visited
            if not nextmask.any():
                break
            tree[nextmask].shape
            v = tree[nextmask]
            v[:, visited] = 0
            # print(m.indices)
            notinf = v != 0
            z = torch.where(notinf, (v * dist[nextmask].unsqueeze(-1)), 0).sum(0)

            dist[notinf.any(0)] = z[notinf.any(0)]
            # print((dist != torch.inf).sum())
            visited |= nextmask
            if visited.all():
                break

    return dists


def bid_on_dists(dists):
    n_fam = dists.shape[0]
    # root_currency_per_node = torch.ones(n_fam, device=dists.device)
    nodes_owned = torch.ones(n_fam, device=dists.device)
    for i in range(100):
        root_currency_per_node = 1 / (nodes_owned + 1)
        bids = root_currency_per_node.unsqueeze(-1) * dists
        bm = bids.max(dim=0)
        idx = bm.indices[bm.values > 0]
        nodes_owned.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float) * 0.03)
        nodes_owned /= 1.03
    return bm


class FamilyGenerator:
    @torch.no_grad()
    def _get_feature_family_treesz(
        self: "Evaluation",
        doc_agg=None,
        threshold=None,
        n=3,
        use_d=False,
        freq_bounds=None,
    ):
        return torch.stack(
            self.generate_feature_families4(
                doc_agg=doc_agg,
                threshold=threshold,
                n=n,
                use_d=use_d,
            )[0]
        )

    @torch.no_grad()
    def _get_feature_family_treeszz3(
        self: "Evaluation",
        doc_agg=None,
        threshold=None,
        use_d=False,
        freq_bounds=None,
    ):
        levels, trees = self.generate_feature_families4(
            doc_agg=doc_agg,
            threshold=threshold,
            use_d=use_d,
        )
        feature_to_family_maps = [{} for _ in levels]
        family_to_feature_maps = [{} for _ in levels]
        d = []
        n = []
        for level, f2fam, fam2f in zip(
            levels, feature_to_family_maps, family_to_feature_maps
        ):
            for i in (level.values != 0).nonzero()[:, 0].tolist():
                fam = level.indices[i].item()
                f2fam[i] = fam
                if fam not in fam2f:
                    fam2f[fam] = []
                fam2f[fam].append(i)

        for l in range(len(levels)):
            for f, feats in family_to_feature_maps[l].items():
                parent_l = l - 1
                parent = 0 if l == 0 else feature_to_family_maps[parent_l][feats[0]]
                self_id = (l, f)
                n.append(
                    {
                        "id": self_id,
                        "level": str(l),
                        "num_features": len(feats),
                        "title": str((self_id, len(feats))),
                        "label": str(self_id),
                    }
                )
                parent_id = (parent_l, parent)
                d.append(
                    {
                        "from": parent_id,
                        "to": self_id,
                    }
                )
        return d, n

    @cache_version(MAIN_VERSION + INFOC_VERSION)
    def _get_feature_families_unlabeled(
        self: "Evaluation", **kwargs
    ) -> GetFamiliesResponse:
        # TODO .cached_call
        levels, trees = self.generate_feature_families4(**kwargs)
        l = [i.indices for i in levels]
        levels = []
        for levelnum, level in enumerate(l):
            fam_ids = level.unique()
            fl = FamilyLevel(
                level=levelnum,
                families={
                    i: Family(
                        level=levelnum,
                        family_id=i,
                        label=None,
                        subfamilies=[],
                        subfeatures=[
                            ScoredFeature(
                                feature=LabeledFeature(
                                    feature_id=int(feat_id),
                                    label=self.get_feature_label(feat_id),
                                ),
                                score=0.9,
                            )
                            for feat_id in (level == fam_id)
                            .nonzero()
                            .flatten()
                            .tolist()
                        ],
                    )
                    for i, fam_id in enumerate(fam_ids)
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

    @cache_version(MAIN_VERSION + INFOC_VERSION)
    def _get_feature_families_unlabeled_old(self, **kwargs) -> GetFamiliesResponse:
        from ..mst import Families, FamilyTreeNode

        # TODO .cached_call
        levels, trees = self.generate_feature_families4(**kwargs)
        levels = trees
        # levels.shape
        famlevels = [Families.from_tree(f) for f in levels]

        niceroots: list[list[FamilyTreeNode]] = [
            [r for r in f.roots]
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
                                feature=LabeledFeature(
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

    def get_feature_families(self: "Evaluation", **kwargs):
        ffs = self.cached_call._get_feature_families_unlabeled(**kwargs)
        for level in ffs.levels:
            for family in level.families.values():
                family.label = self.get_family_label(family)

        return ffs

    def generate_feature_families1(
        self: "Evaluation",
        doc_agg=None,
        threshold=0.1,
        n=3,
        use_d=False,
        freq_bounds=None,
    ):
        if use_d:
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

        from ..mst import mst

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

    def get_C(
        self: "Evaluation", doc_agg, use_d=False, threshold=None, freq_bounds=None
    ):
        if use_d:
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
        return C

    def generate_feature_families2(
        self: "Evaluation",
        doc_agg=None,
        threshold=0.1,
        n=3,
        use_d=False,
        freq_bounds=None,
        min_family_sizes: list[int] | None = None,
    ):
        if min_family_sizes is None:
            min_family_sizes = [20, 12, 7]
        C = self.get_C(
            doc_agg=doc_agg, use_d=use_d, threshold=threshold, freq_bounds=freq_bounds
        )

        from ..mst import Families, mst

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

        from ..mst import my_mst

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

    def generate_feature_families3(
        self: "Evaluation",
        doc_agg=None,
        threshold=0.1,
        n=3,
        use_d=False,
        freq_bounds=None,
        min_family_sizes: list[int] | None = None,
    ):
        if min_family_sizes is None:
            min_family_sizes = [20, 12, 7]
        if use_d:
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

        from ..mst import Families, mst

        levels = []
        families = []
        for i in range(n):
            tree = mst(C).transpose(0, 1)
            roots = ((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0)
            family = Families.from_tree(tree)
            kept = [r for r in family.roots if len(r) > min_family_sizes[i]]
            roots = torch.zeros_like(roots)
            for f in kept:
                roots[f.feature_id] = 1

            dists = distances(tree.cuda(), roots.cuda())
            c = connectedness(tree.cuda() * 0.99, roots.cuda())
            c.shape
            u, n = c.max(dim=0).indices.unique(return_counts=True)

            # dists[roots][torch.arange(roots.sum()), torch.arange(roots.sum())] = 0
            # tree.diag()[:] = 0
            # for i in tqdm.trange(root_ids.shape[0]):
            #     root = root_ids[i]
            #     dist = dists[i]
            #     dist[root] = 0
            #     visited = torch.zeros_like(roots, dtype=torch.bool)
            #     # print("root", root)
            #     while True:
            #         nextmask = (dist != torch.inf) & ~visited
            #         if not nextmask.any():
            #             break
            #         tree[nextmask].shape
            #         v = tree[nextmask]
            #         v[:, visited] = torch.inf
            #         m = v.min(dim=1)
            #         # print(m.indices)
            #         notinf = v != torch.inf
            #         z = torch.where(notinf, (v + dist[nextmask].unsqueeze(-1)), 0).sum(
            #             0
            #         )

            #         dist[notinf.any(0)] = z[notinf.any(0)]
            #         # print((dist != torch.inf).sum())
            #         visited |= nextmask
            #         if visited.all():
            #             break

            dists.min()
            dists[0, 2270]

            tree[533, 159]
            (tree[15808] != torch.inf).nonzero()
            distances(tree.cuda(), roots.cuda())
            connectedness(tree.cuda(), roots.cuda())
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

    def generate_feature_families4(
        self: "Evaluation",
        doc_agg=None,
        threshold=0.0,
        n=3,
        use_d=False,
        min_family_sizes: list[int] | None = None,
        max_num_families: list[int] | None = None,
    ):
        if min_family_sizes is None:
            min_family_sizes = [20, 6, 3]
        if max_num_families is None:
            max_num_families = [2**5, 2**8, 2**11]

        cconn = lambda CC, tree, roots: connectedness(
            tree.to(self.cuda), roots.to(self.cuda)
        ).to(CC.device)
        conn = lambda CC, tree, roots: CC[roots]
        fam_fn = bid_on_dists
        fam_max = lambda x: x.max(dim=0)

        fg = FamilyGen(
            getC=self.cached_call.get_infoC,
            dist_gen=conn,
            fam_fn=bid_on_dists,
            cuda=self.cuda,
            dist_gen_final=conn,
        )

        return fg(
            doc_agg=doc_agg,
            threshold=threshold,
            n=n,
            use_d=use_d,
            min_family_sizes=min_family_sizes,
            max_num_families=max_num_families,
        )

    @cache_version(INFOC_VERSION)
    @torch.no_grad()
    def get_infoC(
        self: "Evaluation",
        doc_agg=None,
        threshold=None,
        use_d=False,
    ):
        unnormalized = self.cached_call.coactivity(doc_agg=doc_agg).cpu()
        if use_d:
            unnormalized = self.cached_call.cosims(doc_agg=doc_agg).cpu()
        unnormalized[unnormalized.isnan()] = 0
        feat_counts = (
            self.doc_activation_counts if doc_agg else self.seq_activation_counts
        ).cpu()

        def denan(x):
            return torch.where(x.isnan() | x.isinf(), torch.zeros_like(x), x)

        def zdiag(x):
            return torch.where(
                torch.eye(x.shape[0], device=x.device, dtype=torch.bool), 0, x
            )

        def isprob(p):
            assert (p >= 0).all() and (p <= 1).all()
            return p

        def probmat(p):
            return isprob(zdiag(denan(p)))

        def ent(p):
            p = isprob(denan(p))
            return torch.where(
                (p > 0) & (p < 1),
                -p * torch.log(p + 1e-6) - (1 - p) * torch.log(1 - p + 1e-6),
                0,
            )

        def nicemat(m):
            return zdiag(denan(m))

        def nent(q, r):
            q, r = nicemat(q), nicemat(r)
            p = nicemat(q / r)
            isprob(p)
            return torch.where(
                (p > 0) & (p < 1),
                -q * torch.log(p) - (r - q) * torch.log(1 - p),
                0,
            )

        num_docs = self.num_docs if doc_agg else self.seq_len * self.num_docs
        a_mat = feat_counts.unsqueeze(1).expand(-1, feat_counts.shape[0])
        b_mat = feat_counts.unsqueeze(0).expand(feat_counts.shape[0], -1)
        v_mat = unnormalized

        p_a = probmat(a_mat / num_docs)
        p_b = probmat(b_mat / num_docs)
        # p_ab = probmat(v_mat / num_docs)
        p_b_given_a = probmat((v_mat + p_b * p_a) / (a_mat + 1))  # +
        # p_b_given_not_a = probmat((b_mat - v_mat + (1 - p_b) * (1 - p_a)) / (num_docs - a_mat + 1))
        # c_mat = p_a * torch.log()
        # info = a_mat * ent(p_b_given_a) + (num_docs - a_mat) * ent(p_b_given_not_a)
        # info = p_a * nent(v_mat, a_mat) + (1 - p_a) * nent((b_mat - v_mat), (num_docs - a_mat)) - nent(b_mat, num_docs)
        # info = (c_val := ent(p_b)) - (
        #     (a_val := p_a * ent(p_b_given_a)) + (b_val := (1 - p_a) * ent(p_b_given_not_a))
        # )
        # info = torch.where((p_b_given_a > p_b), info + 1e-6, 0)
        # (info[p_b_given_a > p_b]).sum()
        # (p_b_given_a > p_b).sum() / (p_b_given_a < p_b).sum()
        # r = p_a * torch.log(p_b_given_a / p_b)
        # r = torch.log(p_b_given_a / p_b)
        r = ent(p_b) - ent(p_b_given_a)

        # t = (1 - p_a) * torch.log(p_b_given_not_a / p_b)
        # other = p_a * torch.log(p_b_given_a / p_b) + (1 - p_a) * torch.log(
        #     p_b_given_not_a / p_b
        # )

        r = denan(r)
        # t = denan(t)
        # r[p_b_given_a > p_b].sum()
        # r[p_b_given_a < p_b].sum()
        # t[p_b_given_a > p_b].sum()
        # t[p_b_given_a < p_b].sum()
        info = r

        # a_mat.max(dim=0, keepdim=True)
        # b_mat.max(dim=0, keepdim=True)
        # c_mat.max(dim=0, keepdim=True)
        # a_mat.max(dim=1, keepdim=True)
        # b_mat.max(dim=1, keepdim=True)
        # c_mat.max(dim=1, keepdim=True)
        # p_b.max()
        # v = b_mat.to(torch.float64) / num_docs
        # v.max()

        info = torch.where((v_mat > 0) & (info > 0), info, 0)

        info = zdiag(denan(info))
        assert (info >= 0).all()
        # learned =
        # c_mat = info.clone()
        c_mat = info
        threshold = threshold if threshold is not None else c_mat[c_mat > 0].median()
        # c_mat = unnormalized / (().cpu().unsqueeze(-1) + 1e-6)

        c_mat.max(dim=0, keepdim=True)
        c_mat.max(dim=1, keepdim=True)

        # c_mat[c_mat.isnan()] = 0
        c_mat[c_mat < threshold] = 0
        return c_mat


from collections.abc import Callable


@define
class FamilyGen:
    getC: Callable
    dist_gen: Callable
    cuda: torch.device
    fam_fn: Callable = bid_on_dists
    transpose_tree: bool = True
    root_connection_mult: float = 0.03
    no_family_connection: float = 0.000001
    dist_gen_final: Callable | None = None

    def nf_fam(self, dist):
        if self.no_family_connection == 0:
            return self.fam_fn(dist)
        dist2 = torch.cat(
            [dist, torch.ones_like(dist[0]).unsqueeze(0) * self.no_family_connection]
        )
        fams = self.fam_fn(dist2)
        m = fams.indices == dist.shape[0]
        fams.values[m] = 0
        fams.indices[m] = 0

        return fams

    def __call__(
        self,
        doc_agg=None,
        threshold=0.0,
        n=3,
        use_d=False,
        min_family_sizes: list[int] | None = None,
        max_num_families: list[int] | None = None,
    ):
        if min_family_sizes is None:
            min_family_sizes = [20, 10, 7, 5, 3]
        if max_num_families is None:
            max_num_families = [2**2, 2**5, 2**8]

        C = self.getC(doc_agg=doc_agg, use_d=use_d, threshold=threshold)
        from ..mst import mst

        levels = []
        roots_l = []
        num_levels = n
        for i in tqdm.trange(n):
            tree = mst(C)
            if self.transpose_tree:
                tree = tree.transpose(0, 1)
            roots = ((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0)
            CC = C.clone()

            CC = CC + CC.transpose(0, 1) * 0.99
            dists = self.dist_gen(CC=CC, tree=tree, roots=roots)
            families = self.nf_fam(dists)
            # maybe add a psuedo family that's connected to everything

            u, n = families.indices[families.values != 0].unique(return_counts=True)
            toosmall = u[n < min_family_sizes[i]]
            roots[roots.nonzero()[toosmall]] = 0
            dists = self.dist_gen(CC=CC, tree=tree, roots=roots)
            families = self.nf_fam(dists)
            u, n = families.indices[families.values != 0].unique(return_counts=True)

            while len(u) > max_num_families[i]:
                remove = n.topk(len(u) - max_num_families[i], largest=False).indices
                roots[roots.nonzero()[u[remove]]] = 0
                dists = self.dist_gen(CC=CC, tree=tree, roots=roots)
                families = self.nf_fam(dists)
                u, n = families.indices[families.values != 0].unique(return_counts=True)
            dists = (self.dist_gen_final or self.dist_gen)(
                CC=CC, tree=tree, roots=roots
            )
            families = self.nf_fam(dists)
            u, n = families.indices[families.values != 0].unique(return_counts=True)

            levels.append(families)
            roots_l.append(roots)
            if i == num_levels - 1:
                break
            mask = torch.zeros_like(CC, dtype=torch.bool)
            for f in tqdm.trange(dists.shape[0]):
                m = (families.indices == f) & (families.values > 0)
                assert m.ndim == 1
                mask |= m.unsqueeze(0) & m.unsqueeze(1)
            C[~mask] = 0
            C[roots] *= self.root_connection_mult
            C[:, roots] *= self.root_connection_mult
        # convert each level to a tree
        trees = []
        for level, roots in zip(levels, roots_l):
            tree = torch.zeros_like(C, dtype=torch.float)
            tree[
                roots.nonzero()[:, 0][level.indices],
                torch.arange(level.indices.shape[0]),
            ] = level.values
            tree.diagonal()[:] = 0
            trees.append(tree)
        return levels, trees
