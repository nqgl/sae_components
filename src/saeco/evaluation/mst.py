import scipy.sparse as ssp
import torch
import tqdm
from attrs import define, field
from torch import Tensor


def my_mst(C):
    il = C.nonzero().t()
    # mil = il[0] > il[1]
    # il = il[:, mil]
    v = C[il.unbind()]
    sorter = v.argsort(descending=True)
    il = il[:, sorter]
    v = v[sorter]
    include = torch.zeros_like(v, dtype=torch.bool)
    rl = torch.ones_like(il, dtype=torch.bool)

    ids = torch.arange(v.shape[0], device=v.device)

    def add(i):
        nonlocal rl
        edge = il[:, i]
        rl[il == edge[0]] = False
        rl[il == edge[1]] = False
        include[i] = True

    def mask():
        return (rl[0] | rl[1]) & (~include)

    def nexti(mask):
        return ids[mask][0]

    include[0] = True
    for _ in tqdm.trange(C.shape[0] - 1):
        m = mask()
        if not m.any():
            break
        i = nexti(m)
        add(i)

    else:
        return il[:, include], v[include]
    return il[:, include], v[include]


def my_mst2(C):
    il = C.nonzero().t()
    mil = il[0] > il[1]
    il = il[:, mil]
    v = C[il.unbind()]
    sorter = v.argsort(descending=True)
    il = il[:, sorter]
    v = v[sorter]
    include = torch.zeros_like(v, dtype=torch.bool)
    rl = torch.ones_like(il, dtype=torch.bool)

    def add(i):
        nonlocal rl, il, v, include
        edge = il[:, i]
        rl[il == edge[0]] = False
        rl[il == edge[1]] = False
        m = mask()
        include[m][0] = True
        rl = rl[:, m]
        il = il[:, m]
        v = v[m]
        # include = include[m]

    def mask():
        return (rl[0] | rl[1]) & (~include)

    def nexti(mask):
        return 0

    include[0] = True
    for _ in tqdm.trange(C.shape[0] - 1):
        m = mask()
        if not m.any():
            break
        i = nexti(m)
        add(i)

    else:
        return il, v[include]
    raise ValueError("Graph not connected")

    # ivc[torch.ones_like(ivc, dtype=torch.bool).triu()] = 0


def mst(C: Tensor):
    C.diag()[:] = 0
    C[C.isnan()] = 0
    adj = C.max() + 0.1
    ivc = torch.where(C > 0, adj - C, 0)
    # ivc[C == 0] = 0
    tree = torch.tensor(
        ssp.csgraph.minimum_spanning_tree(ssp.coo_matrix(ivc)).toarray()
    )
    return torch.where(tree > 0, adj - tree, 0)
    # return tree_re


from functools import cached_property


@define
class FamilyTreeNode:
    feature_id: int
    children: list["FamilyTreeNode"] = field(factory=list)

    def add_child(self, child):
        self.children.append(child)

    @cached_property
    def family(self):
        return set(self.descendant_ids) | {self.feature_id}

    @cached_property
    def children_ids(self):
        return set(child.feature_id for child in self.children)

    @cached_property
    def descendant_ids(self):
        for child in self.children:
            yield child.feature_id
            yield from child.descendant_ids

    def descendants(self):
        for child in self.children:
            yield child
            yield from child.descendants()

    @classmethod
    def from_root_id_and_graph(cls, root_id, graph):
        root = cls(feature_id=root_id)
        for i in graph[root_id].nonzero()[:, 0]:
            root.add_child(cls.from_root_id_and_graph(i, graph))
        return root

    def __len__(self):
        return len(self.family)


@define
class Families:
    roots: list[FamilyTreeNode] = []

    @classmethod
    def from_tree(cls, tree: Tensor):
        root_feats = ((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0)
        families = [
            FamilyTreeNode.from_root_id_and_graph(i, tree)
            for i in tqdm.tqdm(root_feats.nonzero()[:, 0].tolist())
        ]
        families.sort(key=len, reverse=True)
        return cls(roots=families)

    @property
    def root_ids(self):
        return [root.feature_id for root in self.roots]

    def __len__(self):
        return len(self.roots)


# @define
# class Families:
#     roots: list[FamilyTreeNode] = []

#     @classmethod
#     def from_tree(cls, tree: Tensor):
#         root_feats = ((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0)
#         families = [
#             FamilyTreeNode.from_root_id_and_graph(i, tree)
#             for i in tqdm.tqdm(root_feats.nonzero()[:, 0].tolist())
#         ]
#         roots = root_feats.nonzero()[:, 0]
#         tree[root_feats]
#         for r in roots.nonzero()
#         for i in graph[root_id].nonzero()[:, 0]:
#             root.add_child(cls.from_root_id_and_graph(i, graph))
#         return root

#         families.sort(key=len, reverse=True)
#         return cls(roots=families)

#     def __len__(self):
#         return len(self.roots)


#     @classmethod
#     def from_root_id_and_graph(cls, root_id, graph):
# @define
# class Family:
#     root: int
#     children: set = field(default_factory=set)

#     def add_child(self, child):
#         self.children.append(child)

#     @cached_property
#     def family(self):
#         return set(self.descendant_ids()) | {self.feature_id}

#     def descendant_ids(self):
#         for child in self.children:
#             yield child.feature_id
#             yield from child.descendant_ids()

#     def descendants(self):
#         for child in self.children:
#             yield child
#             yield from child.descendants()

#     @classmethod
#     def from_root_id_and_graph(cls, root_id, graph):
#         root = cls(feature_id=root_id)
#         for i in graph[root_id].nonzero()[:, 0]:
#             root.add_child(cls.from_root_id_and_graph(i, graph))
#         return root

#     def __len__(self):
#         return len(self.family)


# @define
# class Families:
#     roots: list[FamilyTreeNode] = []

#     @classmethod
#     def from_tree(cls, tree: Tensor):
#         root_feats = ((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0)
#         families = [
#             FamilyTreeNode.from_root_id_and_graph(i, tree)
#             for i in tqdm.tqdm(root_feats.nonzero()[:, 0].tolist())
#         ]
#         families.sort(key=len, reverse=True)
#         return cls(roots=families)

#     def __len__(self):
#         return len(self.roots)
