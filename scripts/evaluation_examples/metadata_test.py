import torch
from load import ec
from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.named_filter import NamedFilter

torch.backends.cuda.enable_mem_efficient_sdp(False)
# data = ec.metadatas.create("third", torch.bool)


# def value_from_chunk(chunk):
#     return (chunk.tokens_raw.sum(-1) % 3) == 0


# for i, chunk in enumerate(ec.saved_acts.chunks):
#     start = i * ec.saved_acts.cfg.docs_per_chunk
#     end = start + ec.saved_acts.cfg.docs_per_chunk
#     data.tensor[start:end] = value_from_chunk(chunk)
# data.storage.finalize()
metadata = ec.metadatas["third"]
ec.filters["test_filter"] = metadata

filt_eval = ec._apply_filter(ec.filters["test_filter"])

if __name__ == "__main__" and True:

    filt_cosims = filt_eval.cached_call.activation_cosims()
    filt_cosims2 = filt_eval.cached_call.activation_cosims()
    cosims = ec.cached_call.activation_cosims()
    # assert (filt_cosims == filt_evalactivation_cosims.co_occurrence()).all()
    ec.filters["inv_filt"] = ~ec.filters["test_filter"]
    inv_filt_cosims = ec.open_filtered("inv_filt").activation_cosims()
    reg_cosims = ec.activation_cosims()
    # now use this metadata as a filter and then get correlations
    false_filter = torch.zeros_like(metadata)
    # these next two are created as "temporary filtered evaluations"
    # because their filters are unnamed. Thus you can't save associated data to disk
    false_filter_eval = ec._apply_filter(false_filter)
    true_filter_eval = ec._apply_filter(torch.ones_like(metadata))
    true_cosims = true_filter_eval.activation_cosims()
    # eg, these would fail:
    #    true_filter_eval.artifacts["cosims"] = true_cosims
    #    true_filter_eval.cached_call.activation_cosims()
    assert (true_cosims == reg_cosims).all()
    false_cosims = false_filter_eval.activation_cosims()
    # false_cosims = false_filter_eval.masked_activation_cosims()
    filt_mcosims = filt_eval.masked_activation_cosims()
    reg_mcosims = ec.masked_activation_cosims()
    true_mcosims = true_filter_eval.masked_activation_cosims()
    assert false_cosims.isnan().all()
# here's a possibly nice pattern:
ex = filt_eval.top_activating_examples(5, 0.1)
t: torch.Tensor = torch.empty(0)
# feat = filt_eval.features[5]

# v1 = ec.saved_acts.acts[feat.indices()[0]].to_dense()
# v11 = v1[torch.arange(feat.indices().shape[1]), feat.indices()[1], feat.indices()[2]]
# v2 = feat.value.values()
# (v11 == v2.to_dense()).all()
# assert (feat.value.values() == feat.to_dense()[feat.indices()]).all()


def logit_effects(f, filt_eval=filt_eval, **kwargs):
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
    print("    topmax", filt_eval.detokenize(agg.topk(5).indices))
    print("    topmin", filt_eval.detokenize(agg.topk(5, largest=False).indices))
    print("    seqmax", filt_eval.detokenize(res.max(dim=1).indices[:10]))
    print("    seqmin", filt_eval.detokenize(res.min(dim=1).indices[:10]))
    print("  log")
    print("    topmax", filt_eval.detokenize(lagg.topk(5).indices))
    print("    topmin", filt_eval.detokenize(lagg.topk(5, largest=False).indices))
    print("    seqmax", filt_eval.detokenize(lres.max(dim=1).indices[:10]))
    print("    seqmin", filt_eval.detokenize(lres.min(dim=1).indices[:10]))
    print()


def logit_effects2(f, **kwargs):
    logit_effects(f, filt_eval=filt_eval2, **kwargs)


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

    print("    topmax", filt_eval.detokenize(agg.topk(5).indices))
    print("    topmin", filt_eval.detokenize(agg.topk(5, largest=False).indices))
    print("    seqmax", filt_eval.detokenize(res.max(dim=1).indices[:10]))
    print("    seqmin", filt_eval.detokenize(res.min(dim=1).indices[:10]))


def logit_effect_count(f, filt_eval=filt_eval, **kwargs):
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


feat_id = 27
print()
f_i = ec.features[feat_id].indices()
f = torch.zeros_like(filt_eval.saved_acts.data_filter.filter)
doc_num = 2
fi0 = f_i[:, doc_num]
f[fi0[0]] = True
filter2 = NamedFilter(f, "first feat document")
filt_eval2 = Evaluation(
    model_name=ec.model_name,
    saved_acts=ec.saved_acts.filtered(filter2),
    training_runner=ec.training_runner,
    nnsight_model=ec.nnsight_model,
)
p, n = logit_effect_count(feat_id)
p2, n2 = logit_effect_count(feat_id, by_fwad=True)

logit_effects(feat_id, by_fwad=True)

logit_effects(feat_id, scale=0.5, by_fwad=True)
logit_effects(feat_id, scale=0)

logit_effects2(feat_id, by_fwad=True)
logit_effects2(feat_id, scale=1, by_fwad=True)
logit_effects2(feat_id, scale=0)
logit_effects2(feat_id, scale=0.99)

print(ec.detokenize(ec.saved_acts.tokens[fi0[0]]))
ec.detokenize(ec.saved_acts.tokens[fi0[0]][:, fi0[1] :])

f_i[:, 0]


ec: Evaluation
A = ec.open_filtered("filter A")
B = ec.open_filtered("filter B")
co_occurence_delta = A.doc_level_co_occurrence() - B.doc_level_co_occurrence()


A.document_level_pooled_features - B.document_level_pooled_features
