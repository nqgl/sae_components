import torch
from load_comlm_tahoe import root_eval

# Use default config (pooling="max", dose_mode="max", etc.)
# For custom settings, create a config:
# cfg = PerturbationConfig(pooling="mean", dose_mode="slope")
# root.perturbation_config = cfg


def main():
    # Act 1: phenomatching
    ralimetinib = "Ralimetinib dimesylate"
    erlotinib = "Erlotinib"
    drugs = [
        "Ralimetinib dimesylate",
        "Erlotinib",
        "Gefitinib",
        "Osimertinib (mesylate)",
        "PH-797804",
    ]
    drug_metadata_map = root_eval.get_metadata_values_and_strings("drug")
    meta = root_eval.metadata_store["drug"]
    u, c = meta.unique(return_counts=True)
    c
    u
    profiles = root_eval.cached.compute_metadata_effect_profile(
        metadata_map=drug_metadata_map,
    )
    sim = root_eval.compute_metadata_effect_profile(
        metadata_map=drug_metadata_map,
    )
    sim2 = sim.clone()
    sim2 = sim2 / sim2.diag().unsqueeze(0).pow(0.5) / sim2.diag().unsqueeze(1).pow(0.5)
    sim2.diag().sum()
    sim2.diagonal().fill_(0)
    sim2.sum()
    sim2 = sim2.triu()
    sim2 = sim2 / sim2.sum(1, keepdim=True) / sim2.sum(0, keepdim=True)
    m = sim2.max(dim=0)
    print()
    N = 10
    tk = m.values.topk(N)
    for i, j in zip(m.indices[tk.indices], tk.indices, strict=True):
        print(
            f"{drug_metadata_map.value_strings[i]} <--> {drug_metadata_map.value_strings[j]} : {m.values[j]}"
        )

    r_i = drug_metadata_map.value_strings.index(ralimetinib)
    for i in sim2[r_i].topk(10).indices:
        print(drug_metadata_map.value_strings[i])
    top = root_eval.top_similar_drugs(sim, sim_keys, query=ralimetinib, k=5)
    print("Top similar to ralimetinib:")
    for d, s in top:
        print(f"  {d:>12s}: {s:0.4f}")

    # Shared differential features example
    ral = root_eval.cached.compute_drug_profile(ralimetinib)
    erl = root_eval.cached.compute_drug_profile(erlotinib)
    shared = root_eval.top_shared_differential_features(ral, erl, k=10)
    print("\nTop shared features (ralimetinib ↔ erlotinib):")
    for fid, p1, p2, c in shared:
        print(f"  feat {fid:>6d}: p1={p1:+0.19f} p2={p2:+0.19f} contrib={c:+0.19f}")

    # Act 1.3: cytotox (once you pick a candidate)
    CYTOTOX_ID = 42
    token_enrich, logit_effects = root_eval.validate_cytotox_feature(CYTOTOX_ID)
    print("\nToken enrichment output (raw):", type(token_enrich))

    # Act 2: control predictors of sensitivity
    corr = root_eval.cached.compute_feature_sensitivity_correlation(
        drug="ralimetinib",
        response_feature=CYTOTOX_ID,
        cell_lines=None,
    )
    topk = torch.topk(corr, 20)
    print("\nTop control features predicting sensitivity:")
    for fid, r in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
        print(f"  feat {fid:>6d}: r={r:+0.4f}")


if __name__ == "__main__":
    main()
