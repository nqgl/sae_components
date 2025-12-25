import torch
from load import root_eval

from saeco.evaluation.evaluation import Evaluation

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# root_eval.tokenizer.encode(".")
# root_eval.detokenize([13])

if "period_count" not in root_eval.metadatas:
    b = root_eval.metadata_builder(dtype=torch.long, device="cpu")
    for chunk in b:
        b << (chunk.tokens.value == 13).sum(-1)
    root_eval.metadatas["period_count"] = b.value

if "third" not in root_eval.metadatas:
    data = torch.zeros(root_eval.saved_acts.cfg.num_docs, dtype=torch.bool)

    for i, chunk in enumerate(root_eval.saved_acts.chunks):
        start = i * root_eval.saved_acts.cfg.docs_per_chunk
        end = start + root_eval.saved_acts.cfg.docs_per_chunk
        data[start:end] = (chunk.tokens_raw.sum(-1) % 3) == 0
    root_eval.metadatas["third"] = data
metadata = root_eval.metadatas["third"]
if "test_filter" not in root_eval.filters:
    root_eval.filters["test_filter"] = metadata

filtered_eval = root_eval.open_filtered("test_filter")

if __name__ == "__main__" and False:
    filt_cosims = filtered_eval.cached_call.activation_cosims()
    filt_cosims2 = filtered_eval.cached_call.activation_cosims()
    cosims = root_eval.cached_call.activation_cosims()
    # assert (filt_cosims == filt_evalactivation_cosims.co_occurrence()).all()
    # ec.filters["inv_filt"] = ~ec.filters["test_filter"].filter
    inv_filt_cosims = root_eval.open_filtered("inv_filt").activation_cosims()
    reg_cosims = root_eval.activation_cosims()
    # now use this metadata as a filter and then get correlations
    false_filter = torch.zeros_like(metadata)
    # these next two are created as "temporary filtered evaluations"
    # because their filters are unnamed. Thus you can't save associated data to disk
    false_filter_eval = root_eval._apply_filter(false_filter)
    true_filter_eval = root_eval._apply_filter(torch.ones_like(metadata))
    true_cosims = true_filter_eval.activation_cosims()
    # eg, these would fail:
    #    true_filter_eval.artifacts["cosims"] = true_cosims
    #    true_filter_eval.cached_call.activation_cosims()
    assert ((true_cosims == reg_cosims) | reg_cosims.isnan()).all()
    false_cosims = false_filter_eval.activation_cosims()
    # false_cosims = false_filter_eval.masked_activation_cosims()
    filt_mcosims = filtered_eval.masked_activation_cosims()
    reg_mcosims = root_eval.masked_activation_cosims()
    true_mcosims = true_filter_eval.masked_activation_cosims()
    assert false_cosims.isnan().all()
ex = filtered_eval.top_activating_examples(5, 0.1)
# feat = filt_eval.features[5]

# v1 = ec.saved_acts.acts[feat.indices()[0]].to_dense()
# v11 = v1[torch.arange(feat.indices().shape[1]), feat.indices()[1], feat.indices()[2]]
# v2 = feat.value.values()
# (v11 == v2.to_dense()).all()
# assert (feat.value.values() == feat.to_dense()[feat.indices()]).all()


def logit_effects(f, filt_eval=filtered_eval, k=5, **kwargs):
    lres, res = filt_eval.average_patching_effect_on_dataset(f, batch_size=2, **kwargs)
    agg = res.mean(0)
    lagg = lres.mean(0)
    agg = res[0]
    lagg = lres[0]

    if "by_fwad" in kwargs:
        print("fwad")
    else:
        print("patched")
    print("  prob")
    print("    topmax", filt_eval.detokenize(agg.topk(k).indices))
    print("    topmin", filt_eval.detokenize(agg.topk(k, largest=False).indices))
    print("    seqmax", filt_eval.detokenize(res.max(dim=1).indices[:10]))
    print("    seqmin", filt_eval.detokenize(res.min(dim=1).indices[:10]))
    print("  log")
    print("    topmax", filt_eval.detokenize(lagg.topk(k).indices))
    print("    topmin", filt_eval.detokenize(lagg.topk(k, largest=False).indices))
    print("    seqmax", filt_eval.detokenize(lres.max(dim=1).indices[:10]))
    print("    seqmin", filt_eval.detokenize(lres.min(dim=1).indices[:10]))
    print()


def compare_for_overlap(f, filt_eval=filtered_eval, k=20, **kwargs):
    lres, res = filt_eval.average_patching_effect_on_dataset(f, batch_size=2, **kwargs)
    lresf, resf = filt_eval.average_patching_effect_on_dataset(
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
                    print(i, j, ii, filt_eval.detokenize([ii.item()]))

    return t


def aggregator(ec: Evaluation):
    pos_counts = torch.zeros(ec.seq_len, ec.d_vocab).cuda()
    # neg_counts = torch.zeros(ec.seq_len, ec.d_vocab).cuda()

    def process(batch, batch_seq_positions):
        for i in range(batch.shape[0]):
            b, bsp = batch[i], batch_seq_positions[i]
            pos_counts[:-bsp] += b[bsp:] > 0
            # neg_counts[:-bsp] += b[bsp:] < 0

    return pos_counts, process


def print_effects(res):
    agg = res.mean(0)
    agg = res[0]

    print("    topmax", filtered_eval.detokenize(agg.topk(5).indices))
    print("    topmin", filtered_eval.detokenize(agg.topk(5, largest=False).indices))
    print("    seqmax", filtered_eval.detokenize(res.max(dim=1).indices[:10]))
    print("    seqmin", filtered_eval.detokenize(res.min(dim=1).indices[:10]))


def logit_effect_count(f, filt_eval=filtered_eval, **kwargs):
    p, proc = aggregator(filt_eval)

    p2, proc2 = aggregator(filt_eval)
    num_batches = filt_eval.custom_patching_effect_aggregation(
        f, proc, proc2, batch_size=2, **kwargs
    )
    print("logitspace")
    print_effects(p)
    print("probspace")
    print_effects(p2)
    return p, p2


if True:

    def logit_effects2(f, **kwargs):
        logit_effects(f, filt_eval=filt_eval2, **kwargs)

    feat_id = 44
    print()
    f_i = root_eval.features[feat_id].indices()
    f = torch.zeros_like(filtered_eval.saved_acts.data_filter.filter)
    doc_num = 2
    fi0 = f_i[:, doc_num]
    f[fi0[0]] = True
    # filter2 = NamedFilter(f, "first feat document")
    filt_eval2 = root_eval._apply_filter(f)
    p, n = logit_effect_count(feat_id, random_subset_n=10)
    p2, n2 = logit_effect_count(feat_id, by_fwad=True, random_subset_n=10)
    logit_effects(feat_id, by_fwad=True, random_subset_n=100)

    logit_effects(feat_id, scale=0.5)
    logit_effects(feat_id, by_fwad=True, k=20)

    t = compare_for_overlap(feat_id, k=100)
    logit_effects2(feat_id, by_fwad=True)
    logit_effects2(feat_id, scale=1, by_fwad=True)
    logit_effects2(feat_id, scale=0)
    logit_effects2(feat_id, scale=0.99)

    print(root_eval.detokenize(root_eval.saved_acts.tokens[fi0[0]]))
    root_eval.detokenize(root_eval.saved_acts.tokens[fi0[0]][:, fi0[1] :])

    f_i[:, 0]

filtered_eval.features[41].indices()

root_eval: Evaluation
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


diff = A.acts_avg_over_dataset("mean", "mean") - B.acts_avg_over_dataset("mean", "mean")
