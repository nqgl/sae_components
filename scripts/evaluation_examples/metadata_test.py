import torch
from load import ec
from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.filtered_evaluation import NamedFilter
from saeco.evaluation.metadata import MetaDatas

# data = ec.metadatas.create_metadata("third", torch.bool)


# def value_from_chunk(chunk):
#     return (chunk.tokens_raw.sum(-1) % 3) == 0


# for i, chunk in enumerate(ec.saved_acts.chunks):
#     start = i * ec.saved_acts.cfg.docs_per_chunk
#     end = start + ec.saved_acts.cfg.docs_per_chunk
#     data.storage.tensor[start:end] = value_from_chunk(chunk)
# data.storage.finalize()
metadata = ec.metadatas["third"]
filt = NamedFilter(metadata.tensor, "test_filter")
inv_filt = NamedFilter(~metadata.tensor, "test_inv_filter")

filt_eval = Evaluation(
    model_name=ec.model_name,
    saved_acts=ec.saved_acts.filtered(filt),
    training_runner=ec.training_runner,
    nnsight_model=ec.nnsight_model,
)

inv_filt_eval = Evaluation(
    model_name=ec.model_name,
    saved_acts=ec.saved_acts.filtered(inv_filt),
    training_runner=ec.training_runner,
    nnsight_model=ec.nnsight_model,
)
if __name__ == "__main__" and False:

    filt_cosims = filt_eval.activation_cosims()
    # assert (filt_cosims == filt_eval.co_occurrence()).all()
    inv_filt_cosims = inv_filt_eval.activation_cosims()
    reg_cosims = ec.activation_cosims()
    # now use this metadata as a filter and then get correlations
    true_filter = NamedFilter(
        torch.ones_like(metadata.storage.tensor), "all_true_filter"
    )
    true_filter_eval = Evaluation(
        model_name=ec.model_name,
        saved_acts=ec.saved_acts.filtered(true_filter),
        training_runner=ec.training_runner,
        nnsight_model=ec.nnsight_model,
    )
    true_cosims = true_filter_eval.activation_cosims()
    assert (true_cosims == reg_cosims).all()
    false_filter = NamedFilter(
        torch.zeros_like(metadata.storage.tensor), "all_false_filter"
    )
    false_filter_eval = Evaluation(
        model_name=ec.model_name,
        saved_acts=ec.saved_acts.filtered(false_filter),
        training_runner=ec.training_runner,
        nnsight_model=ec.nnsight_model,
    )
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


def logit_effects(f, **kwargs):
    lres, res = filt_eval.patching_effect_on_dataset(f, batch_size=8, **kwargs)
    agg = res.mean(0)
    lagg = lres.mean(0)
    print(filt_eval.detokenize(agg.topk(5).indices))
    print(filt_eval.detokenize(agg.topk(5, largest=False).indices))
    print(filt_eval.detokenize(res.max(dim=1).indices[:10]))
    print(filt_eval.detokenize(res.min(dim=1).indices[:10]))
    print(filt_eval.detokenize(lagg.topk(5).indices))
    print(filt_eval.detokenize(lagg.topk(5, largest=False).indices))
    print(filt_eval.detokenize(lres.max(dim=1).indices[:10]))
    print(filt_eval.detokenize(lres.min(dim=1).indices[:10]))


def fwad_logit_effects(f, **kwargs):
    res = filt_eval.avg_fwad_effect_on_dataset(f, batch_size=8, **kwargs)
    agg = res.mean(0)
    print(filt_eval.detokenize(agg.topk(5).indices))
    print(filt_eval.detokenize(agg.topk(5, largest=False).indices))
    print(filt_eval.detokenize(res.max(dim=1).indices[:10]))
    print(filt_eval.detokenize(res.min(dim=1).indices[:10]))


feat_id = 15
fwad_logit_effects(feat_id)

fwad_logit_effects(feat_id, scale=0.5)
logit_effects(feat_id, scale=0.9999)
print()
f_i = ec.features[feat_id].indices()
f = torch.zeros_like(filt_eval.saved_acts.data_filter.filter)
fi0 = f_i[:, 0]
f[fi0[0]] = True
filter2 = NamedFilter(f, "first feat document")
filt_eval2 = Evaluation(
    model_name=ec.model_name,
    saved_acts=ec.saved_acts.filtered(filter2),
    training_runner=ec.training_runner,
    nnsight_model=ec.nnsight_model,
)


def logit_effects(f, **kwargs):
    lres, res = filt_eval2.patching_effect_on_dataset(f, batch_size=8, **kwargs)
    agg = res.mean(0)
    lagg = lres.mean(0)
    print(filt_eval2.detokenize(agg.topk(5).indices))
    print(filt_eval2.detokenize(agg.topk(5, largest=False).indices))
    print(filt_eval2.detokenize(res.max(dim=1).indices[:10]))
    print(filt_eval2.detokenize(res.min(dim=1).indices[:10]))
    # print(filt_eval2.detokenize(lagg.topk(5).indices))
    # print(filt_eval2.detokenize(lagg.topk(5, largest=False).indices))
    # print(filt_eval2.detokenize(lres.max(dim=1).indices[:10]))
    # print(filt_eval2.detokenize(lres.min(dim=1).indices[:10]))


def fwad_logit_effects(f, **kwargs):
    res = filt_eval2.avg_fwad_effect_on_dataset(f, batch_size=8, **kwargs)
    agg = res.mean(0)
    print(filt_eval2.detokenize(agg.topk(5).indices))
    print(filt_eval2.detokenize(agg.topk(5, largest=False).indices))
    print(filt_eval2.detokenize(res.max(dim=1).indices[:10]))
    print(filt_eval2.detokenize(res.min(dim=1).indices[:10]))


fwad_logit_effects(feat_id)
fwad_logit_effects(feat_id, scale=1)
logit_effects(feat_id, scale=0)
logit_effects(feat_id, scale=0.99)

ec.detokenize(ec.saved_acts.tokens[fi0[0]])
ec.detokenize(ec.saved_acts.tokens[fi0[0]][fi0[1] - 1 :])

f_i[:, 0]
