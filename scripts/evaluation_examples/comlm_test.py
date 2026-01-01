# %%
from collections.abc import Callable

import torch
from comlm.datasource.training_batch import NoisedBatch
from comlm.exprank import XRNoisedBatch
from load_comlm_tahoe import root_eval
from torch import Tensor

from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig
from saeco.data.config.model_config.model_cfg import ModelConfig
from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.fastapi_models.EnrichmentSortBy import EnrichmentSortBy

NoisedBatch
root: Evaluation[XRNoisedBatch] = root_eval


model_cfg: ModelConfig[ComlmModelConfig] = root.sae_cfg.train_cfg.data_cfg.model_cfg  # type: ignore

arch = model_cfg.model_load_cfg.pretrained_arch
comlm_cfg = arch.run_cfg
# %%

arch.data.metadata_tokenizer
# %%
root.features[0].value.device
# %%
root.features[1].indices()
# %%
metadata_tokenizer = arch.data.metadata_tokenizer
for key in comlm_cfg.arch_cfg.metadata_embedding_config.selected_metadata:
    tokenizer = metadata_tokenizer.tokenizers[key]


# %%
"""
maybe we will want some "compare intervention on input data" method

Also, I should probably drop out either all or none of the metadata for 

"""


data0 = root.samples[0:4]
# %%

root.get_inputs_type()
# %%
root.saved_acts.get_inputs_type()
# %%
#
# Initialize metadata if necessary
#
metadata_keys = list(metadata_tokenizer.tokenizers.keys())
has_metadata = [key in root.metadata_store for key in metadata_keys]
if not all(has_metadata):
    assert not any(has_metadata), "Some metadata keys are already present"
    all_metadata_builder = root.metadata_builder(
        dtype=torch.long,
        device="cpu",
        item_size=(len(metadata_keys),),
    )
    for chunk in all_metadata_builder:
        all_metadata_builder << chunk.tokens.value["metadata"]
    for i, key in enumerate(metadata_keys):
        tokenizer = metadata_tokenizer.tokenizers[key]
        metadata = all_metadata_builder.value[:, i]
        root.metadata_store[key] = metadata
        root.metadata_store.set_str_translator(
            key, {"<<PAD>>": 0, "<<UNK>>": 1, **tokenizer.tokens}
        )
# %%
ac = root.activation_cosims(out_device="cpu", blocks_per_dim=2)

# %%
# root.d_dict**2 / 1e9
# %%

causal_coacts = root.causal_coacts(out_device="cpu", f_chunk_i=8192, f_chunk_j=2048)

# %%
# act_coacts = root.act_coacts2(out_device="cpu", f_chunk_i=8192, f_chunk_j=2048)

# %%
causal_coacts.numel()
# %%
(causal_coacts > 1).sum() / 4294967296
# %%
c = (causal_coacts > 0).sum(1)
# %%
c.topk(10, largest=False)

# %%
cs = causal_coacts.sum(1)
# %%
cs.topk(10)
# %%
cs0 = causal_coacts.sum(0)
# %%
cs0.topk(10)
# %%

# def _feature_activity_sum(self) -> Tensor:
#     activity = torch.zeros(self.d_dict, dtype=torch.float).to(self.cuda)
#     for chunk in self.saved_acts.chunks:
#         acts = chunk.acts.value.to(self.cuda).to_dense()
#         activity += acts.sum(dim=1).sum(dim=0)
#     return activity
base_freqs = root._feature_activity_sum()
base_freqs.topk(10), base_freqs.shape
# %%


def tset(x):
    return set(int(i) for i in x)


@torch.inference_mode()
def normalize_coacts(coacts: Tensor) -> Tensor:
    base_rate = root
    diag = coacts.diag().sqrt() + 1e-9
    return coacts / diag.unsqueeze(0) / diag.unsqueeze(1)


N = 10
tset(cs0.topk(N).indices) & tset(cs.topk(N).indices)

# %%
# root.cuda = torch.device("cpu")

root.seq_activation_counts
# %%
root.doc_activation_counts
# %%
ccd = causal_coacts.diag().sqrt() + 1e-6
# %%
cc = causal_coacts / ccd.unsqueeze(0) / ccd.unsqueeze(1)
# %%
N = 200
tset(cc.sum(0).topk(N).indices) & tset(cc.sum(1).topk(N).indices)
# %%
tset(root.seq_activation_counts.topk(N).indices) & tset(
    cc.sum(0).topk(N).indices
) & tset(cc.sum(1).topk(N).indices)
# %%
tset(root.doc_activation_counts.topk(N).indices) & tset(
    cc.sum(0).topk(N).indices
) & tset(cc.sum(1).topk(N).indices)
# %%
ccnm = root.causal_coacts(
    out_device="cpu",
    acts_pre_mod_func=lambda acts: (acts > 0).float(),
    f_chunk_i=8192,
    f_chunk_j=2048,
)

# %%
ccnm
ccm = normalize_coacts(causal_coacts)
ccn = normalize_coacts(ccnm)

# %%
cm0 = ccm.sum(0)
cm1 = ccm.sum(1)
cn0 = ccn.sum(0)
cn1 = ccn.sum(1)
# %%

N = 10


def feature_overlaps(N: int, **kwargs: Tensor):
    keys = {
        tuple(sorted((k0, k1)))
        for k0 in kwargs.keys()
        for k1 in kwargs.keys()
        if k0 != k1
    }
    for k0, k1 in keys:
        v0 = kwargs[k0]
        v1 = kwargs[k1]
        overlap = tset(v0.topk(N).indices) & tset(v1.topk(N).indices)
        print(f"{k0} & {k1}: {overlap}")


feature_overlaps(N, cm0=cm0, cm1=cm1, cn0=cn0, cn1=cn1)
# %%
interesting_maybe = tset(cm1.topk(N).indices) - tset(cn1.topk(N).indices)
interesting_maybe = tset(cm0.topk(N).indices) - tset(cn0.topk(N).indices)

interesting_maybe
# %%
for k in interesting_maybe:
    print(k)
    f = root.get_feature(k)
    print(
        f.top_activations(k=10).top_activations_metadata_enrichments(
            metadata_keys=metadata_keys
        )
    )
    # break
# %%
bool_metadatas = [
    "is_blood_related",
    "is_cancer",
    "is_cancer_cell_line",
    "is_chronic_disease",
    "is_control",
    "is_drug_treated",
    "is_genetic_perturb",
    "is_healthy",
    "is_immune_cells",
    "is_infectious",
    "is_perturbed",
    "is_primary_cells",
    "is_stem_cells",
]
good_metadatas = [
    "cell_prep",
    # "disease_group",
    # "disease_weight",
    # "entrez_id",
    "experiment_type",
    # "lib_prep",
    # "log_median_genes",
    # "log_median_umi",
    # "median_genes_per_cell",
    # "median_umi_per_cell",
    # "obs_count",
    # "organism",
    # "overall_quality_percentile",
    # "overall_weight",
    # "passes_qc",
    # "passes_qc_stringent",
    "perturbation_detailed",
    # "perturbation_type",
    # "perturbation_weight",
    # "srx_accession",
    # "tech_10x",
    "disease_category",
    "disease_standardized",
    "disease_detailed",
    "tissue",
    "cell_line_subcategory",
    "cell_line_type_category",
    # "tissue_expanded",
    # "tissue_weight",
]


def print_for_metadatas(metadata_keys: list[str], num_top_meta: int = 3):
    for mk, data in (
        f.top_activations(p=0.1)
        .top_activations_metadata_enrichments(metadata_keys=metadata_keys)
        .items()
    ):
        print(mk)
        d = data.remove_zero_counts().sort(sort_by=EnrichmentSortBy.score)
        # data.labels
        labeled = root.metadata_store.translate({d.name: d.labels})[d.name]
        for label, proportion, score, normalized_count, count in zip(
            labeled[:num_top_meta],
            d[:num_top_meta].proportions,
            d[:num_top_meta].scores,
            d[:num_top_meta].normalized_counts,
            d[:num_top_meta].counts,
            strict=True,
        ):
            print(f"\t{label}: {score} | {normalized_count} | ({count})")
        print("\t...")
        for label, proportion, score, normalized_count, count in zip(
            labeled[-num_top_meta:],
            d[-num_top_meta:].proportions,
            d[-num_top_meta:].scores,
            d[-num_top_meta:].normalized_counts,
            d[-num_top_meta:].counts,
            strict=True,
        ):
            print(f"\t{label}: {score} | {normalized_count} | ({count})")

    # print(f"d[:num_top_meta].labels: {labeled[:num_top_meta]}")
    # print(f"d[:num_top_meta].proportions: {d[:num_top_meta].proportions}")
    # print(f"d[:num_top_meta].scores: {d[:num_top_meta].scores}")
    # print(f"d[:num_top_meta].normalized_counts: {d[:num_top_meta].normalized_counts}")
    # print(f"d[:num_top_meta].counts: {d[:num_top_meta].counts}")


print_for_metadatas(good_metadatas, 3)

# %%
print_for_metadatas(bool_metadatas, 3)

# %%


def score_enrichment(
    total_counts: torch.Tensor,
    total_denom: int | float,
    counts: torch.Tensor,
    sel_denom: torch.Tensor | float | int,
    smoothing=1,
    r=0,
    base_smoothing=1,
    r_base=0,
):
    assert total_counts.ndim == counts.ndim == 1
    base_rate = total_counts.float().mean() / total_denom
    # print(f"base_rate: {base_rate}")
    p_total = (total_counts + base_smoothing * base_rate**r_base) / (
        total_denom + base_smoothing * base_rate ** (r_base - 1)
    )
    # print(f"base_rate - p_total: {base_rate - p_total}")
    p_subset = (counts + smoothing * p_total**r) / (
        sel_denom + smoothing * p_total ** (r - 1)
    )
    return torch.log2(p_subset / p_total)


# %%
import saeco.evaluation.return_objects as enr

# %%
enr.score_enrichment = score_enrichment
# %%

print_for_metadatas(good_metadatas, 3)
# %%
base_freqs = base_freqs.cpu()
mean_acts = base_freqs / root.num_samples
ccb = causal_coacts / mean_acts.unsqueeze(0) / mean_acts.unsqueeze(1)
# %%
ccb.diag()
# %
c0b = torch.layer_norm(causal_coacts, causal_coacts.shape[1:], eps=1e-6)

# %%
interesting_maybe = tset(c0b.abs().sum(0).topk(10).indices)
# %%
interesting_maybe & tset(cm0.topk(10).indices)
# %%
for i in interesting_maybe:
    f = root.get_feature(i)
    print_for_metadatas(good_metadatas, 3)
    input()
# %%
# root.get_acts_with_intervention(feature=interesting_maybe[0])
model = root.nnsight_model
# for layer in root.nnsight_model.layers:
batch = root.samples[:4]
args, kwargs = (
    root.sae_cfg.train_cfg.data_cfg.model_cfg.model_load_cfg.unpack_model_inputs(
        batch, {}
    )
)


def make_modified_kwargs(rank_mod_fn: Callable[[Tensor], Tensor]):
    r = kwargs["ranks"].clone()  # .float()
    kw2 = kwargs.copy()
    kw2["ranks"] = rank_mod_fn(r)
    return kw2


kw2 = make_modified_kwargs(lambda r: r)


def modify_rank_fn(r: Tensor):
    # z = torch.zeros_like(r)
    # z[:, 10] += 2
    r[:, 2] = 0
    return r


kw3 = make_modified_kwargs(modify_rank_fn)
# %%
with torch.inference_mode():
    r = arch.model(*args, **kwargs)
    r2 = arch.model(*args, **kw2)
    l = r.logits
    l2 = r2.logits
    r3 = arch.model(*args, **kw3)
    l3 = r3.logits

with torch.inference_mode():
    diff = l - l2
    diff3 = l - l3
diff.abs().max(), diff3.abs().max()
diff.abs().mean(), diff3.abs().mean()

diff3.abs().mean(-1).max(-1)
# %%

zeros = 0
total = 0

for i in range(50):

    def modify_rank_fn(r: Tensor):
        # z = torch.zeros_like(r)
        # z[:, 10] += 2
        r[:, i] = 0
        return r

    kw3 = make_modified_kwargs(modify_rank_fn)
    with torch.inference_mode():
        r = arch.model(*args, **kwargs)
        r2 = arch.model(*args, **kw2)
        l = r.logits
        l2 = r2.logits
        r3 = arch.model(*args, **kw3)
        l3 = r3.logits

    with torch.inference_mode():
        diff = l - l2
        diff3 = l - l3
    diff.abs().max(), diff3.abs().max()
    diff.abs().mean(), diff3.abs().mean()

    diff3.abs().mean(-1).max(-1)
# %%
mx = diff[0].abs().max(dim=1)
tk = mx.values.topk(10)
tk

# %%
l[0, 835, 0], l2[0, 835, 0]
# diff[0,835,0]
l[0, 835, 0] + 0.1
# %%
with model.trace(*args, **kwargs):
    (x,), xk = model.layers[0].attn.f.rope.inputs
    x_saved = x.save()
    ranks = xk["ranks"]
    ranks_saved = ranks.save()
# %%
ranks_saved

with model.trace(*args, **kw2):
    (x,), xk = model.layers[0].attn.f.rope.inputs
    x_saved = x.save()
    ranks = xk["ranks"]
    ranks_saved = ranks.save()
# %%


# %%
