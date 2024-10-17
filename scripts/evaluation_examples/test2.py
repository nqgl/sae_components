import sys

import torch
from load import root_eval


# t = root_eval.cached_call.test()
# t2 = root_eval.cached_call.test()
# t3 = root_eval.test()
# print(t, t2, t3)


def f(): ...


setattr(f, "_version", 1)


def dists(ls):
    return [[len(root) for root in level.roots] for level in ls]


def above(ls, n):
    return [[root for root in level.roots if len(root) > n] for level in ls]


def nabove(ls, n):
    return [len(level) for level in above(ls, n)]


def dist_counter(ls):
    for level in ls:
        print([len(root) for root in level.roots[:50]])


root_eval.artifacts.keys()
ff = root_eval.get_feature_families()
ff.levels[0].families[4].subfamilies
from saeco.evaluation.fastapi_models.families_draft import FamilyRef

ff.levels[0].families[4]

res = root_eval.top_activations_and_metadatas_for_family(
    ff.levels[0].families[0], k=100, return_str_docs=True
)

for d in res[0]:
    print(" ".join(d))
fff = [
    root_eval.cached_call._get_feature_family_trees(
        doc_agg="count", freq_bounds=(0, 0.5)
    ),
    root_eval.cached_call._get_feature_family_trees(doc_agg="count"),
    root_eval.cached_call._get_feature_family_trees(),
    root_eval.cached_call._get_feature_family_trees(freq_bounds=(0, 0.1)),
]
from saeco.evaluation.mst import Families

famss = [[Families.from_tree(f) for f in fam] for fam in fff]
[[len([r for r in f.roots if len(r) > 10]) for f in fam if len(f) > 5] for fam in famss]
dist_counter(fams)
print([len(f) for f in fams])
elevels2 = root_eval.generate_feature_families(threshold=0)

f = Families.from_tree(levels2[0])
len(f.roots[1])
f2 = Families.from_tree(levels2[1])
len(f2.roots[4])
f3 = Families.from_tree(levels2[2])

910

len(f3.roots[0])
fams = [f, f2, f3]
dist_counter(fams)
f.roots[0].feature_id
root_eval.seq_activation_probs.topk(100)
[f.roots[0].feature_id]

r00 = f.roots[0]
len(r00)
for r in f2.roots:
    print((len(r) - 1) / len(r.children))
print()


print(len(levels2[0].roots[0]), len(levels2[1].roots[0]), len(levels2[2].roots[0]))
print(len(levels2[0]), len(levels2[1]), len(levels2[2]))
nabove(levels2, 4)
dist_counter(levels2)
root_eval.seq_activation_counts[levels2[0].roots[3].feature_id]
root_eval.num_docs * 128
root_eval.num_docs * 128 / 3

seq_hyper = (root_eval.seq_activation_counts).topk(5)
doc_hyper = (root_eval.doc_activation_counts).topk(5)
doc_hyper
seq_hyper

out = root_eval.top_activations_token_enrichments(
    feature=seq_hyper.indices[0].item(), k=100
)
out[3]


def family_active(family, threshold=0.1):
    v = None
    for i in family:
        feat = root_eval.features[i]
        t = feat.value > threshold
        if v is None:
            v = t
        else:
            v = v | t


# levels4

levels4 = root_eval.generate_feature_families(doc_agg="count", threshold=0.0)
print(len(levels4[0].roots[0]), len(levels4[1].roots[0]), len(levels4[2].roots[0]))
print(len(levels4[0]), len(levels4[1]), len(levels4[2]))

levels1 = root_eval.generate_feature_families("max", threshold=0)
print(len(levels1[0].roots[0]), len(levels1[1].roots[0]), len(levels1[2].roots[0]))
print(len(levels1[0]), len(levels1[1]), len(levels1[2]))


levels3 = root_eval.generate_feature_families("max", threshold=0.9)
print(len(levels3[0].roots[0]), len(levels3[1].roots[0]), len(levels3[2].roots[0]))
print(len(levels3[0]), len(levels3[1]), len(levels3[2]))


def count_above(n):
    for levels in [levels1, levels2, levels3, levels4]:
        print()
        for level in levels:
            print(len([root for root in level.roots if len(root) > n]))


count_above(5)


dist_counter(levels4)
len(levels4[1].roots[0].family)
f1 = levels4[1].roots[0].family
f2 = levels4[1].roots[2].family
s1 = {int(i) for i in f1}
s2 = {int(i) for i in f2}
len(f1)
len(f2)
len(s1)
len(s2)
len(s1 & s2) / len(s1 | s2)
[f for f in f1 if isinstance(f, int)]
len(levels4[0])
len(levels4[0].roots[0])
len({int(i) for i in levels4[0].roots[0].family})
levels4[0].roots[0].feature_id
feat_counts = root_eval.cached_call._feature_num_active_docs().to(root_eval.cuda)
feat_counts[levels4[0].roots[0].feature_id]
feat_counts.float().mean()
(feat_counts > 5_000).sum()
root_eval.num_docs
