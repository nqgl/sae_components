import torch
from load import ec
from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.filtered_evaluation import Filter
from saeco.evaluation.metadata import MetaDatas

# data = ec.metadatas.create_metadata("test4", torch.bool)


# def value_from_chunk(chunk):
#     return (chunk.tokens == 123).any(-1)


# for i, chunk in enumerate(ec.saved_acts.chunks):
#     start = i * ec.saved_acts.cfg.docs_per_chunk
#     end = start + ec.saved_acts.cfg.docs_per_chunk
#     data.storage.tensor[start:end] = value_from_chunk(chunk)
# data.storage.finalize()
metadata = ec.metadatas["test4"]
print(metadata.storage.tensor.nonzero())
filt = Filter(metadata.storage.tensor, "test_filter")
filt_eval = Evaluation(
    model_name=ec.model_name,
    saved_acts=ec.saved_acts.filtered(filt),
    training_runner=ec.training_runner,
    nnsight_model=ec.nnsight_model,
)

filt_cosims = filt_eval.activation_cosims()
reg_cosims = ec.activation_cosims()
# now use this metadata as a filter and then get correlations


# here's a possibly nice pattern:
t: torch.Tensor = torch.empty(0)
for piece, chunk in zip(t.chunk(ec.cache_cfg.num_chunks), ec.saved_acts.chunks):
    ...
    # operate on the piece and the chunk.whatever
