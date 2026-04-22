# %%

import torch
from comlm.datasource.training_batch import NoisedBatch
from comlm.exprank import XRNoisedBatch
from load_comlm_tahoe import root_eval
from torch import Tensor

from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig
from saeco.data.config.model_config.model_cfg import ModelConfig
from saeco.evaluation.eval_components.perturbation_analysis import MetadataValueMap
from saeco.evaluation.evaluation import Evaluation

NoisedBatch
root: Evaluation[XRNoisedBatch] = root_eval


model_cfg: ModelConfig[ComlmModelConfig] = root.sae_cfg.train_cfg.data_cfg.model_cfg  # type: ignore

arch = model_cfg.model_load_cfg.pretrained_arch
comlm_cfg = arch.run_cfg

metadata_tokenizer = arch.data.metadata_tokenizer
# %%
# Initialize metadata if necessary
#
metadata_keys = list(metadata_tokenizer.tokenizers.keys())
has_metadata = [key in root.metadata_store for key in metadata_keys]
has_metadata
# %%
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


ralimetinib = "Ralimetinib dimesylate"
CONTROL_DRUG = "DMSO_TF"
drug_metadata_map = root_eval.get_metadata_values_and_strings("drug")


def translate_drugs(drugs: list[str]):
    return [
        e
        for (e,) in [
            [
                d
                for i, d in enumerate(drug_metadata_map.value_strings)
                if drug.lower() in d.lower()
            ]
            for drug in drugs
        ]
    ]


ctrl_index = drug_metadata_map.value_strings.index(CONTROL_DRUG)
egfrs = translate_drugs(["erlotinib", "gefitinib", "osimertinib"])
p3as = translate_drugs(["PH-797804"])

# %%


def print_from_sim(sim: Tensor, metadata_map: MetadataValueMap):
    # sim2 = sim.clone()
    # sim2 = sim2 / sim2.diag().unsqueeze(0).pow(0.5) / sim2.diag().unsqueeze(1).pow(0.5)
    # sim2.diag().sum()
    # sim2.diagonal().fill_(0)
    # sim2.sum()
    # sim2 = sim2.triu()
    # sim2 = sim2 / sim2.sum(1, keepdim=True) / sim2.sum(0, keepdim=True)
    sim = sim.triu(diagonal=1)
    m = sim.max(dim=0)
    print()
    N = 10
    tk = m.values.topk(N)
    for i, j in zip(m.indices[tk.indices], tk.indices, strict=True):
        print(
            f"{metadata_map.value_strings[i]} <--> {metadata_map.value_strings[j]} : {m.values[j]}"
        )

    print("\nsimilar to ralimetinib:")

    r_i = metadata_map.value_strings.index(ralimetinib)
    for i in sim[r_i].topk(30).indices:
        s = metadata_map.value_strings[i]
        if any(d in s.lower() for d in egfrs):
            print(f"EGFR: {s}")
        elif any(d in s.lower() for d in ["PH-797804"]):
            print(f"p38a: {s}")
        else:
            print(s)


def print_ral_scores(sim: Tensor, metadata_map: MetadataValueMap):
    r_i = metadata_map.value_strings.index(ralimetinib)
    print("------------")
    print("EGFR SCORES:")
    for e in egfrs:
        e_i = metadata_map.value_strings.index(e)
        print(f"{e}: {sim[r_i, e_i]}")
    print("------------")
    print("p38a SCORES:")
    for p in p3as:
        p_i = metadata_map.value_strings.index(p)
        print(f"{p}: {sim[r_i, p_i]}")
    print("------------")


def get_similarity(profile: Tensor):
    sim = profile @ profile.T
    d = sim.diag().add(1e-9).pow(0.5)
    return (sim / d.unsqueeze(0) / d.unsqueeze(1)).triu(diagonal=1)


with torch.inference_mode():
    for pooling in ("max", "sum"):
        profiles = root_eval.cached.compute_metadata_effect_profile(
            metadata_map=drug_metadata_map,
            pooling=pooling,
        )
        sim = get_similarity(profiles)
        diff = profiles - profiles[ctrl_index].unsqueeze(0)
        sim_diff = get_similarity(diff)
        print(
            f"\n----------------------------\nPooling: {pooling}"
            "\n----------------------------\n"
        )
        print("\n\nplain similarities\n-------------")
        print_ral_scores(sim, drug_metadata_map)
        # print_from_sim(sim, drug_metadata_map)
        print("\n\ndiff similarities:\n-------------")
        print_ral_scores(sim_diff, drug_metadata_map)
        # print_from_sim(sim_diff, drug_metadata_map)

# sim =
# sim2 = sim.clone()
# sim2 = sim2 / sim2.diag().unsqueeze(0).pow(0.5) / sim2.diag().unsqueeze(1).pow(0.5)
# sim2.diag().sum()
# sim2.diagonal().fill_(0)
# sim2.sum()
# sim2 = sim2.triu()
# sim2 = sim2 / sim2.sum(1, keepdim=True) / sim2.sum(0, keepdim=True)
# m = sim2.max(dim=0)
# print()
# N = 10
# tk = m.values.topk(N)
# for i, j in zip(m.indices[tk.indices], tk.indices, strict=True):
#     print(
#         f"{drug_metadata_map.value_strings[i]} <--> {drug_metadata_map.value_strings[j]} : {m.values[j]}"
#     )

# r_i = drug_metadata_map.value_strings.index(ralimetinib)
# for i in sim2[r_i].topk(10).indices:
#     print(drug_metadata_map.value_strings[i])
# top = root_eval.top_similar_drugs(sim, sim_keys, query=ralimetinib, k=5)


# # %%

# # %%
# profiles.shape
# # %%
# norms = profiles.norm(dim=1)

# norms.min(), norms.max()


# # %%
# norms.topk(10)

# # %%
# feature_norms = profiles.norm(dim=0)
# # %%

# feature_norms.min(), feature_norms.max()
# # %%
# fnormed_profiles = profiles / (feature_norms.unsqueeze(0) + 1e-6)
# # %%
# fn_norms = fnormed_profiles.norm(dim=1)
# # %%
# fn_norms.min(), fn_norms.max()
# # %%
# fn_norms.topk(10, largest=False)
# # %%
# naive_diffs = profiles - profiles[ctrl_index].unsqueeze(0)

# # %%
# sims = naive_diffs @ naive_diffs.T
# # %%
# sim = sims.triu(diagonal=1)
# # %%

# # %%
# print_from_sim(sim, drug_metadata_map)
# # %%
# diffs2 = fnormed_profiles - fnormed_profiles[ctrl_index].unsqueeze(0)

# # %%
# sim2 = (diffs2 @ diffs2.T).triu(diagonal=1)

# print_from_sim(sim2, drug_metadata_map)
# # %%

# %%
