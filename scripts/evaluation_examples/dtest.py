
import pandas as pd
import torch
from load import root_eval

# all_families = root_eval.get_feature_families()
# fams2 = [all_families.levels[0].families[0], all_families.levels[0].families[1]]
# res = root_eval.top_overlapped_feature_family_documents(
#     families=fams2, k=100, return_str_docs=True
# )
# root_eval


def logit_effects_df(f, **kwargs):
    k = 30
    lres, res = root_eval.average_patching_effect_on_dataset(f, batch_size=1, **kwargs)
    d = {}

    def add(name, data):
        maxk = data.topk(k)
        mink = data.topk(k, largest=False)

        for topk, mm in [[maxk, "max"], [mink, "min"]]:
            d[f"{name}_{mm}_tokens"] = root_eval.detokenize(topk.indices)
            d[f"{name}_{mm}_values"] = topk.values.cpu()

    add("prob", res.mean(0))
    add("log", lres.mean(0))
    for i in range(10):
        add(f"prob_{i}", res[i])
        add(f"log_{i}", lres[i])
    return pd.DataFrame(d)


def logit_effects(f, k=5, **kwargs):
    lres = root_eval.average_patching_effect_on_dataset(f, batch_size=1, **kwargs)
    lagg = lres.mean(0)
    lagg = lres[0]

    if "by_fwad" in kwargs:
        print("fwad")
    else:
        print("patched")
    print("    topmax", root_eval.detokenize(lagg.topk(k).indices))
    print("    topmin", root_eval.detokenize(lagg.topk(k, largest=False).indices))
    print("    seqmax", root_eval.detokenize(lres.max(dim=1).indices[:10]))
    print("    seqmin", root_eval.detokenize(lres.min(dim=1).indices[:10]))
    print()


def compare_for_overlap(f, root_eval=root_eval, k=20, **kwargs):
    lres, res = root_eval.average_patching_effect_on_dataset(f, batch_size=2, **kwargs)
    lresf, resf = root_eval.average_patching_effect_on_dataset(
        f, batch_size=2, by_fwad=True, **kwargs
    )
    t = torch.tensor(
        [
            # res[0].topk(k).indices.tolist(),
            # resf[0].topk(k).indices.tolist(),
            # res[0].topk(k, largest=False).indices.tolist(),
            # resf[0].topk(k, largest=False).indices.tolist(),
            # lres[0].topk(k).indices.tolist(),
            lresf[0].topk(k).indices.tolist(),
            lres[0].topk(k).indices.tolist(),
            # lresf[0].topk(k, largest=False).indices.tolist(),
        ],
        dtype=torch.long,
    )
    for i in range(len(t)):
        for j in range(i):
            for ii in t[i]:
                if (ii == t[j]).any():
                    print(i, j, ii, root_eval.detokenize([ii.item()]))

    return t


def print_effects(res):
    agg = res.mean(0)
    agg = res[0]

    print("    topmax", root_eval.detokenize(agg.topk(5).indices))
    print("    topmin", root_eval.detokenize(agg.topk(5, largest=False).indices))
    print("    seqmax", root_eval.detokenize(res.max(dim=1).indices[:10]))
    print("    seqmin", root_eval.detokenize(res.min(dim=1).indices[:10]))


if __name__ == "__main__":
    feat_id = 42
    logit_effects(feat_id)
    logit_effects(feat_id, by_fwad=True)
    logit_effects(feat_id, by_fwad=2)
    logit_effects(feat_id, by_fwad=3)

    t = compare_for_overlap(feat_id, k=100)
    dff = logit_effects_df(feat_id, by_fwad=True)
    dfd = logit_effects_df(feat_id)

    if True:
        feat_id = 42
        logit_effects(feat_id, random_subset_n=100)
        input("Press Enter to continue...")
        logit_effects(feat_id, by_fwad=True, random_subset_n=100)

        logit_effects(feat_id, scale=0.5)
        logit_effects(feat_id, by_fwad=True, k=20)

        t = compare_for_overlap(feat_id, k=100)

        logit_effects(feat_id, by_fwad=True)
        logit_effects(feat_id, scale=1, by_fwad=True)
        logit_effects(feat_id, scale=0)
        logit_effects(feat_id, scale=0.99)

        # print(root_eval.detokenize(root_eval.saved_acts.tokens[fi0[0]]))
        # root_eval.detokenize(root_eval.saved_acts.tokens[fi0[0]][:, fi0[1] :])

    root_eval.features[41].indices()

    root_eval.detokenize([9999])
    spend = 9999
    root_eval.detokenize([9989])
    android = 9989
    builder = root_eval.metadata_builder(torch.bool, "cpu")
    for chunk in builder:
        builder << (chunk.tokens.value == spend).any(-1)
    root_eval.filters["filter A"] = builder.value
    builder = root_eval.metadata_builder(torch.bool, "cpu")
    for chunk in builder:
        builder << (chunk.tokens.value == android).any(-1)
    root_eval.filters["filter B"] = builder.value
    A = root_eval.open_filtered("filter A")
    B = root_eval.open_filtered("filter B")
    co_occurence_delta = A.doc_level_co_occurrence() - B.doc_level_co_occurrence()

    diff = A.acts_avg_over_dataset("mean", "mean") - B.acts_avg_over_dataset(
        "mean", "mean"
    )
