import torch

from saeco.evaluation import Evaluation

MODEL_NAME = "sae sweeps/dynamic_thresh_sae0.001[50.0]-10384/50001"  # relative path from ~/workspace/saved_models/ to the SAE, minus the extensions
STORAGE_NAME = "stored_acts"

### Setup:

# ## First time only:
# e = Evaluation.from_model_name(MODEL_NAME)
# e.store_acts(
#     CachingConfig(
#         dirname=STORAGE_NAME,
#         num_chunks=10,
#         docs_per_chunk=300,
#         documents_per_micro_batch=16,
#     ),
#     displace_existing=True,
# )


root = Evaluation.from_cache_name(STORAGE_NAME)


# Retrieving a feature:
FEAT_NUM = 0
feature_tensor = root.features[FEAT_NUM]  # features are FilteredTensor objects
# shape is (num_docs, sequence_length, num_features)
# in this case, we are operating on the root evaluation object,
# so the filter does not filter out any documents. But this helps to
# make behavior consistent between filtered and root evaluations.

# Retrieving the active documents for a feature:
feature_indices = feature_tensor.indices()
# feature_indices is a 3 by N tensor, where N is the number of active documents
# feature_indices[0] are document indices
# feature_indices[1] are sequence position indices
# feature_indices[2] are feature indices. Since these are all from the same feature:
assert (feature_indices[2] == FEAT_NUM).all()


# we can also get the top activating examples:
top_tensor = root.top_activating_examples(FEAT_NUM, p=0.1)  # top 10% of activations
top_indices = top_tensor.indices()

top_docs = root.docs[top_indices[0]]

i = 0
print(root.detokenize(top_docs[i]))
print("active on:", root.detokenize(top_docs[i][top_indices[1][i].item()]))

# if there are specific use cases you have let me know and I can add support for them
# eg if it would be good to have an iterator that iterates over active docs and gives the
# text and activations


### metadata, filters, artifacts
# these are all stored to disk


# metadata and filters associate some extra info with each document

# - filters specifically are bool tensors that can be used to get a
# filtered subset of the data
# - whereas metadata can be any dtype and has more freedom in shape
# - artifacts can be any shape/dtype and are separated by which filter you are using
#    - this is so computations that are expensive can be saved to disk and reused

### making metadatas
# it's just a tensor, so eg a metadata that is the document id mod 4 would be:
if "mod 4" not in root.metadatas:
    root.metadatas["mod 4"] = torch.arange(root.num_docs) % 4

# feature active:
if "feature active" not in root.metadatas:
    root.metadatas["feature active"] = (
        root.features[FEAT_NUM].to_dense().value.count_nonzero(-1).bool()
    )


# making less trivial metadatas that actually care about the contents of the data is different:
# eg, to create a metadata that includes documents with a certain token:
SELECTED_TOKEN_ID = 9999
if "selected token" not in root.metadatas:
    builder = root.metadata_builder(dtype=torch.bool, device="cpu")
    for chunk in builder:
        builder << (chunk.tokens.value == SELECTED_TOKEN_ID).any(-1)
    root.metadatas["selected token"] = builder.value
# we do it with this loop because we've got more data than fits in memory,
# so it's stored in chunks and we're iterating through the chunks

# if we wanted to create a metadata that counts the average number of active features

if "average active features" not in root.metadatas:
    builder = root.metadata_builder(torch.float, "cpu")
    for chunk in builder:
        builder << chunk.acts.value.to_dense().count_nonzero(-1).float().mean(dim=-1)
    root.metadatas["average active features"] = builder.value


# filters are just boolean tensors that can be used to filter the data
# eg,
if "very active" not in root.filters:
    root.filters["very active"] = root.metadatas["average active features"] > 50

# if there is a filter, you can now open a filtered evaluation object
filtered_eval = root.open_filtered("very active")


### other less organized stuff:

# getting the metadatas of the max-activating examples
top_tensor = root.top_activating_examples(FEAT_NUM, p=0.1)
metadata = root.metadatas["average active features"]
top_tensor_filtered = top_tensor.filter_inactive_docs()
filtered_metadata = top_tensor_filtered.to_filtered_like_self(
    metadata,
    presliced=False,
    premasked=False,
    ndim=1,  # might change this
)

# simpler but only works on root Evaluations
top_tensor = root.top_activating_examples(FEAT_NUM, p=0.1)
metadata = root.metadatas["average active features"]
filtered_metadata2 = top_tensor.filter_inactive_docs().filter.apply_mask(metadata)
assert (filtered_metadata.value == filtered_metadata2).all()

# notes on filteredtensor
# it represents a larger tensor
# has slices, mask, and value
# on some FilteredTensor ft, those are attributes:
# ft.filter.slices, ft.filter.mask, ft.value
# it is representing (part of) some larger tensor t where:
# t.shape == ft.shape
# t[ft.filter.slices][ft.filter.mask] == ft.value
# It does not at this moment support actual torch ops, but helps w/ keeping
# filtering and value in sync
# I might implement full support for torch ops in the future,
# especially easier if it's only a subset that are needed.
# oh also has the nice property that
# ft.filter.slice(t)[ft.filter.mask] = x
# or
# ft.filter.writeat(t, x)
# will write x to the appropriate location because of the way slice then and mask works
# however if you try to do (the following or anything analagous):
# ft.filter.slice(t)[ft.filter.mask][:] = x it won't be able to write to t anymore.
# can explain further if needed
print("done")
