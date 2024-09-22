import torch
from load import ec
from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.filtered_evaluation import NamedFilter
from saeco.evaluation.metadata import MetaDatas

# data = ec.metadatas.create_metadata("test4", torch.bool)


# def value_from_chunk(chunk):
#     return (chunk.tokens_raw == 123).any(-1)


# for i, chunk in enumerate(ec.saved_acts.chunks):
#     start = i * ec.saved_acts.cfg.docs_per_chunk
#     end = start + ec.saved_acts.cfg.docs_per_chunk
#     data.storage.tensor[start:end] = value_from_chunk(chunk)
# data.storage.finalize()
metadata = ec.metadatas["test4"]
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


filt_cosims = filt_eval.activation_cosims()
# assert (filt_cosims == filt_eval.co_occurrence()).all()
inv_filt_cosims = inv_filt_eval.activation_cosims()
reg_cosims = ec.activation_cosims()
# now use this metadata as a filter and then get correlations
true_filter = NamedFilter(torch.ones_like(metadata.storage.tensor), "all_true_filter")
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
t: torch.Tensor = torch.empty(0)
for piece, chunk in zip(t.chunk(ec.cache_cfg.num_chunks), ec.saved_acts.chunks):
    ...
    # operate on the piece and the chunk.whatever
