import sys

import jaal
import pandas as pd

import torch
from load import root_eval

f = root_eval.cached_call._get_feature_families_unlabeled()
all_families = root_eval.get_feature_families()
families = [v for level in all_families.levels for k, v in level.families.items()]
root_eval.init_family_psuedofeature_tensors(families)

print()
f.levels[0].families[3].subfamilies

f.levels[1].families[8].subfamilies
f.levels[2].families[21].subfeatures

# 0,3 -> 1,8 seems ok maybe
# 1,8 -> 2,21 not ok


l, t = root_eval.generate_feature_families4()

feat2fams = []

len(l)
for level in l:
    fams = level.indices
    fams[level.values == 0] = -1
    feat2fams.append(fams)
fams2feat = []
for level in feat2fams:
    fam2feat = []
    for i in range(level.max() + 1):
        fam2feat.append(set((level == i).nonzero()[:, 0].tolist()))
    fams2feat.append(fam2feat)
import networkx as nx
from pyvis.network import Network

i = -1

for i in range(len(fams2feat[0])):
    f1 = fams2feat[1][i]
    for j in range(len(fams2feat[1])):
        f2 = fams2feat[2][j]
        if f2 & f1:
            if f2 - f1:
                print(f2 - f1)
                print("wrong")
            else:
                print(i, j)

for i in range(len(fams2feat[1])):
    f1 = fams2feat[1][i]
    for j in range(len(fams2feat[2])):
        f2 = fams2feat[2][j]
        if f2 & f1:
            if f2 - f1:
                print(f2 - f1)
                print("wrong")
            else:
                print(i, j)

# all_families = root_eval.get_feature_families()
# llens = [len(l.families) for l in all_families.levels]
# flid = [sum([0] + llens[:j]) for j in range(len(llens))]
# d = []
# nodes = []
# for l, level in enumerate(all_families.levels):
#     for f, family in level.families.items():
#         self_id = (l, f)
#         for sf in family.subfamilies:
#             d.append(
#                 {
#                     "from": self_id,
#                     "to": (l + 1, sf.family.family_id),
#                     "score": sf.score,
#                 }
#             )
#         nodes.append(
#             {
#                 "id": self_id,
#                 "level": str(l),
#                 "num_features": (
#                     len(family.subfeatures) if l == 2 else len(family.subfamilies)
#                 ),
#                 "title": len(family.subfeatures),
#             }
#         )

# jaal.Jaal(edge_df=pd.DataFrame(d), node_df=pd.DataFrame(nodes)).plot()\
levels = l
feature_to_family_maps = [{} for _ in levels]
family_to_feature_maps = [{} for _ in levels]
d = []
n = []
for level, f2fam, fam2f in zip(levels, feature_to_family_maps, family_to_feature_maps):
    for i in (level.values != 0).nonzero()[:, 0].tolist():
        fam = level.indices[i].item()
        f2fam[i] = fam
        if fam not in fam2f:
            fam2f[fam] = []
        fam2f[fam].append(i)
# for i in range(self.d_dict):
#     for j in range(1,3):

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
jaal.Jaal(edge_df=pd.DataFrame(d), node_df=pd.DataFrame(n)).plot()
