import sys

import torch
from load import root_eval

sys.setrecursionlimit(100_000)


def dists(ls):
    return [[len(root) for root in level.roots] for level in ls]


def above(ls, n):
    return [[root for root in level.roots if len(root) > n] for level in ls]


def nabove(ls, n):
    return [len(level) for level in above(ls, n)]


def dist_counter(ls):
    for level in ls:
        print([len(root) for root in level.roots[:50]])


levels2 = root_eval.generate_feature_families(threshold=0.1)
print(len(levels2[0].roots[0]), len(levels2[1].roots[0]), len(levels2[2].roots[0]))
print(len(levels2[0]), len(levels2[1]), len(levels2[2]))
nabove(levels2, 4)
dist_counter(levels2)
root_eval.seq_activation_counts[levels2[0].roots[3].feature_id]
root_eval.num_docs * 128
root_eval.num_docs * 128 / 3

seq_hyper = (root_eval.seq_activation_counts).topk(5)
doc_hyper


def family_active(family, threshold=0.1):
    v = None
    for i in family:
        feat = root_eval.features[i]
        t = feat.value > threshold
        if v is None:
            v = t
        else:
            v = v | t


levels4

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
