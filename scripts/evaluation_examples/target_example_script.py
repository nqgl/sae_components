# %% [markdown]
# # Ralimetinib Demo - Implementation Spec v3
#
# Spec for Glen. References his existing code, identifies gaps.
#
# ## Data Structure
#
# | cell_line | drug        | dosage | feature_activations           |
# |-----------|-------------|--------|-------------------------------|
# | LU65      | DMSO        | 0      | [s0 control]                  |
# | LU65      | ralimetinib | 1,2,3  | [s1 perturbed at 3 doses]     |
# | SW48      | DMSO        | 0      | [s0 control]                  |
# | SW48      | ralimetinib | 1,2,3  | [s1 perturbed at 3 doses]     |
#
# ~380 drugs × 50-100 cell lines × 3 dosages + controls
#
# ## Key Cell Lines
# - **SW48**: EGFR G674S mutation, KRAS-WT (sensitive)
# - **LU65**: EGFR L858R + KRAS G12C (resistant - KRAS bypass)
# - Various KRAS-mutant lines (negative controls)
#
# ## Key Drugs
# - **ralimetinib**: labeled p38α inhibitor, actually EGFR
# - **PH-797804, doramapimod**: actual p38α inhibitors (negative controls)
# - **erlotinib, gefitinib, osimertinib**: known EGFR inhibitors (should phenomatch)

# %% [markdown]
# ---
# # Run of Show
#
# ## ACT 1: Drug-Drug Phenomatching
# **Goal:** Show ralimetinib clusters with EGFR inhibitors, not p38 inhibitors.
#
# **Core idea:** Each drug creates a "perturbation signature" - how it shifts the cell's feature
# activations from control (s0, DMSO) to perturbed (s1, drug-treated). If two drugs have similar
# signatures, they likely share mechanism of action. "Clusters with" = high cosine similarity
# of these perturbation vectors.
#
# **The key computation:**
# ```
# delta[drug, cell_line, dosage] = features(s1_perturbed) - features(s0_control)
# ```
# A delta is specific to a (drug, cell_line, dosage) triplet. For one drug, we get a 3D tensor.
#
# - **Step 1.1** - Core Primitives (THE MAIN THING)
#   - `compute_drug_deltas_tensor(drug)` → (n_cell_lines, n_dosages, n_features) ← full 3D
#   - `compute_drug_deltas_matrix(drug)` → (n_cell_lines, n_features) after dose reduction ← **CORE BUILDING BLOCK**
#
# - **Step 1.2** - Drug Similarity
#   - `compute_drug_similarity_matrix(drugs, mode="profile"|"pattern")`
#   - `top_similar_drugs()`, `top_shared_differential_features()`
#
# - **Step 1.3** - Find Cytotoxicity Feature (uses existing APIs)
#   - We need a feature that indicates "drug is killing cells"
#   - This lets us measure drug-sensitivity: sensitive cells die (high cytotox), resistant cells survive (low cytotox)
#   - Validate with: token enrichment, GO enrichment, lambdit effects
#   - Also validate with PRISM data (Adam handles this): PRISM is external cell viability data
#     showing how much each drug kills each cell line. Our cytotox feature should correlate with it.
#
# **Expected:** Ralimetinib ↔ EGFR inhibitors (r≈0.8), NOT p38 inhibitors (r≈0.2)
#
# ---
# ## ACT 2: Feature-Dependence Biomarkers
# **Goal:** Show s0 EGFR features predict drug sensitivity.
#
# **Why are some cells sensitive and others resistant?**
# - Ralimetinib is actually an EGFR inhibitor (we showed this in Act 1)
# - Cells with high EGFR pathway activation are DEPENDENT on EGFR for survival
# - Block EGFR → they die → SENSITIVE (e.g., SW48)
# - Cells with KRAS mutations have a bypass: KRAS keeps them alive even without EGFR
# - Block EGFR → KRAS compensates → RESISTANT (e.g., LU65)
#
# **Hypothesis:** s0 EGFR feature activation predicts sensitivity (indicates EGFR-dependence)
#
# **How we measure sensitivity:**
# The cytotox feature (from Act 1) = signature of cell death.
# For each cell line, measure the INCREASE in cytotox feature activation when drug is applied:
# ```
# sensitivity["SW48"] = cytotox_activation(SW48 + drug) - cytotox_activation(SW48 + DMSO) = 0.8
#                       ↑ big increase in cell death signature → drug kills → SENSITIVE
#
# sensitivity["LU65"] = cytotox_activation(LU65 + drug) - cytotox_activation(LU65 + DMSO) = 0.2
#                       ↑ small increase → drug doesn't kill → RESISTANT
# ```
# This gives us ONE NUMBER per cell line.
# Now we ask: which features in the CONTROL state (s0, before drug) predict this?
# I.e., look at feature activations in DMSO-treated cells, averaged per cell line.
#
# - `compute_sensitivity_by_cell_line()` - reuses `compute_drug_deltas_matrix`
# - `compute_feature_sensitivity_correlation()` - Pearson r per feature
# - **Alternative:** attach sensitivity metadata to controls → reuse your `find_predictive_features()`
#
# **Extension: Mutation-stratified analysis**
# - Zoom in on EGFR-mutant cell lines specifically
# - Compare s0 features between EGFR-activating (sensitive) vs KRAS (resistant) mutations
# - Negative controls: KRAS mutations confer resistance, MAPK mutations have no effect
# - Note: we don't have T790M resistance mutation cell line
#
# **Expected:** EGFR features r≈0.6, p38 features r≈0
#
# ---
# ## ACT 3: Mechanistic Attribution
# **Goal:** Feature→feature circuits: SW48 (EGFR→cytotox) vs LU65 (KRAS→survival bypass)
#
# **Status:** Depends on PR #28
#
# ---
# ## Open Questions
# 1. Token aggregation: max vs mean across gene positions? (the `pooling` param)
# 2. Feature normalization: z-score before similarity?
# 3. Cell aggregation: mean activations vs pseudobulk? (the `cell_agg` param)
#    - Individual cells are noisy (~2k UMI per cell)
#    - "mean_acts": average feature activations across cells in condition
#    - "pseudobulk": aggregate raw expression first, run through model once
# 4. Fold change vs raw delta? Could normalize: `delta / s0_activation` ("feature activation fold change")

# %% [markdown]
# ---
# # ACT 1: Drug-Drug Phenomatching
#
# **Goal:** Show ralimetinib clusters with EGFR inhibitors, not p38 inhibitors.

# %% [markdown]
# ## Step 1.1: Core Primitives - Perturbation Deltas
#
# For a drug, compute the full 3D tensor: (n_cell_lines, n_dosages, n_features)
#
# Each entry: `delta[cl, dose, :] = mean(perturbed[cl, dose]) - mean(control[cl])`
#
# Then reduce along dosage dimension (max dose or slope) → (n_cell_lines, n_features)
#
# **Implementation approach** (builds on your `compute_feature_activity_per_doc`):
# 1. Call `compute_feature_activity_per_doc(root)` ONCE → (num_docs, d_dict)
# 2. Build masks for each (cell_line, dosage) combo and control
# 3. Subtract control from each perturbed condition

# %%
def compute_drug_deltas_tensor(
    root,
    drug: str,
    cell_lines: "list[str]",
    pooling: str = "max",
    cell_agg: str = "mean_acts",
) -> "torch.Tensor":
    """
    Compute full perturbation delta tensor for a drug.
    Returns: (n_cell_lines, 3, n_features)  ← 3 drug dosages, NOT including control

    Each entry is already a DELTA from control:
        result[cl, dose, :] = features(drug @ dose) - features(DMSO)

    Control (DMSO) is used to compute deltas but not stored - it's the same
    for all drugs, so no need to repeat it.

    cell_agg: How to aggregate across cells in a condition (individual cells are noisy!)
        "mean_acts": Run each cell through model, average the feature activations
        "pseudobulk": Aggregate raw expression first, run once through model

    Storage: ~100 cell_lines × 3 doses × 65k features ≈ 78MB per drug.
    Fine for one drug at a time; don't cache all 380 drugs.

    Sketch (cell_agg="mean_acts"):
        feature_activity = compute_feature_activity_per_doc(root, pooling=pooling)
        # → (num_docs, d_dict)

        drug_meta = root._root_metadatas["drug"].to("cuda")
        cell_meta = root._root_metadatas["cell_line"].to("cuda")
        dose_meta = root._root_metadatas["dosage"].to("cuda")

        result = zeros(n_cell_lines, 3, n_features)
        for i, cl in enumerate(cell_lines):
            control_mask = (drug_meta == DMSO_ID) & (cell_meta == cl_id)
            control_acts = feature_activity[control_mask].mean(dim=0)  # avg across cells

            for j, dose in enumerate([1, 2, 3]):
                perturbed_mask = (drug_meta == drug_id) & (cell_meta == cl_id) & (dose_meta == dose)
                perturbed_acts = feature_activity[perturbed_mask].mean(dim=0)  # avg across cells
                result[i, j, :] = perturbed_acts - control_acts

        return result

    Sketch (cell_agg="pseudobulk"):
        # Aggregate raw expression per condition FIRST, then run through model once
        # This may require different data loading - TBD
    """
    raise NotImplementedError


def compute_drug_deltas_matrix(
    root,
    drug: str,
    cell_lines: "list[str]",
    pooling: str = "max",
    dose_mode: str = "max",
    cell_agg: str = "mean_acts",
) -> "torch.Tensor":
    """
    Compute perturbation deltas for a drug, reduced along dosage dimension.
    Returns: (n_cell_lines, n_features)

    dose_mode:
        "max": Take highest dose → deltas[:, -1, :]
        "slope": Fit dose-response slope along dim=1

    cell_agg: see compute_drug_deltas_tensor

    CORE BUILDING BLOCK for both Act 1 and Act 2.
    """
    deltas_3d = compute_drug_deltas_tensor(root, drug, cell_lines, pooling, cell_agg)
    # → (n_cell_lines, 3, n_features)

    if dose_mode == "max":
        return deltas_3d[:, -1, :]  # highest dose
    elif dose_mode == "slope":
        # fit slope along dosage dimension
        raise NotImplementedError
    raise ValueError(f"Unknown dose_mode: {dose_mode}")


# %% [markdown]
# ## Step 1.2: Drug-Drug Similarity
#
# Two approaches:
# - **profile**: aggregate across cell lines → (n_features,) per drug → cosine sim
# - **pattern**: keep full (n_cell_lines, n_features) matrix → compare patterns
#
# Pattern mode is richer - captures which cell lines respond similarly.


# %%
def compute_drug_profile(
    root,
    drug: str,
    cell_lines: "list[str]",
    pooling: str = "max",
    dose_mode: str = "max",
    cell_agg: str = "mean_acts",
    aggregation: str = "mean",
    normalize: bool = True,
) -> "torch.Tensor":
    """Aggregate perturbation delta across cell lines. Returns: (n_features,)"""
    matrix = compute_drug_deltas_matrix(
        root, drug, cell_lines, pooling, dose_mode, cell_agg
    )
    # → (n_cell_lines, n_features)

    if normalize:
        matrix = (matrix - matrix.mean(0)) / (matrix.std(0) + 1e-8)

    if aggregation == "mean":
        return matrix.mean(0)
    elif aggregation == "median":
        return matrix.median(0).values
    raise ValueError(f"Unknown aggregation: {aggregation}")


def compute_drug_similarity_matrix(
    root,
    drugs: "list[str]",
    cell_lines: "list[str]",
    mode: str = "profile",  # "profile" or "pattern"
    pooling: str = "max",
    dose_mode: str = "max",
    cell_agg: str = "mean_acts",
) -> "torch.Tensor":
    """
    Returns: similarity_matrix (n_drugs, n_drugs)

    mode="profile": aggregate first, compare (n_features,) vectors
    mode="pattern": compare full (n_cell_lines, n_features) matrices
    """
    if mode == "profile":
        profiles = [
            compute_drug_profile(root, d, cell_lines, pooling, dose_mode, cell_agg)
            for d in drugs
        ]
        profiles = torch.stack(profiles)
        normed = profiles / profiles.norm(dim=1, keepdim=True)
        return normed @ normed.T

    elif mode == "pattern":
        matrices = [
            compute_drug_deltas_matrix(
                root, d, cell_lines, pooling, dose_mode, cell_agg
            )
            for d in drugs
        ]
        flattened = torch.stack([m.flatten() for m in matrices])
        normed = flattened / flattened.norm(dim=1, keepdim=True)
        return normed @ normed.T

    raise ValueError(f"Unknown mode: {mode}")


def top_similar_drugs(sim_matrix, drugs, query: str, k: int = 10):
    """Get k most similar drugs to query."""
    idx = drugs.index(query)
    sims = sim_matrix[idx]
    top_idx = sims.argsort(descending=True)[1 : k + 1]
    return [(drugs[i], sims[i].item()) for i in top_idx]


def top_shared_differential_features(
    profiles, drugs, drug1: str, drug2: str, k: int = 5
):
    """Which features have the most similar deltas between two drugs?"""
    p1 = profiles[drugs.index(drug1)]
    p2 = profiles[drugs.index(drug2)]
    contribution = p1 * p2  # positive if same direction, large if both large
    top_idx = contribution.argsort(descending=True)[:k]
    return [(int(i), p1[i].item(), p2[i].item()) for i in top_idx]

    # TODO: Normalize by feature's baseline variance across all cells.
    # Raw deltas are misleading - a +0.5 delta in a low-variance feature is more
    # meaningful than +0.5 in a high-variance feature.
    #
    # Option: z-score each feature's delta by its std across all cells:
    #     feature_stds = compute_feature_stds_across_all_cells(root)  # (n_features,)
    #     p1_normed = p1 / feature_stds
    #     p2_normed = p2 / feature_stds
    #     contribution = p1_normed * p2_normed


# %% [markdown]
# ## Step 1.3: Find the Cytotoxicity Feature
#
# Before phenomatching, identify a feature that captures cell death.
# Uses **existing APIs** - no new code needed.
#
# **Validation criteria:**
# 1. Token enrichment shows apoptosis genes (CASP3, BAX, etc.)
# 2. GO enrichment for "apoptotic process" (existing)
# 3. Lambdit effects: ↑CASP3, ↑BAX, ↓BCL2, ↓MCL1
# 4. PRISM correlation (see Appendix - Adam handles this)


# %%
def validate_cytotox_feature(root, feature_id: int):
    """Validate a candidate cytotox feature using existing APIs."""
    # Token enrichment - look for apoptosis genes
    token_enrich = root.top_activations_token_enrichments(
        feature=feature_id,
        k=500,
        mode="active",
    )

    # Lambdit effects - should promote death genes, suppress survival
    logit_effects = root.average_aggregated_patching_effect_on_dataset(
        feature_id=feature_id,
        random_subset_n=500,
    )

    return token_enrich, logit_effects


# %% [markdown]
# ## Act 1 Expected Output
#
# ```
# Top drugs similar to ralimetinib:
#   1. erlotinib (EGFR):     0.847
#   2. gefitinib (EGFR):     0.823
#   3. osimertinib (EGFR):   0.801
#   ...
#   43. PH-797804 (p38α):    0.203  <-- NEGATIVE CONTROL
#   44. doramapimod (p38α):  0.187  <-- NEGATIVE CONTROL
#
# Top shared effects (ralimetinib ↔ erlotinib) - features with similar deltas:
#   Feature 1234 (EGFR_signaling):  Δ=+0.72 / Δ=+0.68  (both drugs increase this)
#   Feature 5678 (MAPK_cascade):    Δ=+0.54 / Δ=+0.61  (both drugs increase this)
#   Feature 4321 (cytotox_program): Δ=+0.48 / Δ=+0.52  (both drugs trigger apoptosis)
# ```
#
# **Key insight:** Ralimetinib clusters with EGFR inhibitors, NOT p38α inhibitors.
# The similarity is driven by shared effects on EGFR pathway features.
#
# **Narrative order:** First show known EGFR inhibitors cluster together (erlotinib, gefitinib, etc.),
# THEN reveal ralimetinib follows the same pattern. Don't lead with ralimetinib.
#
# **Sanity check:** Confirm s0 baselines don't already differ in cytotox feature across cell lines.
# The cytotox difference should come from drug treatment, not baseline variation.

# %% [markdown]
# ---
# # ACT 2: Feature-Dependence Biomarkers
#
# **Goal:** Show EGFR pathway activation in CONTROL state predicts drug sensitivity.
#
# See Run of Show for full explanation of:
# - Why some cells are sensitive vs resistant (EGFR-dependence vs KRAS bypass)
# - How we measure sensitivity (cytotox delta per cell line)
# - What we're predicting (s0 features → drug response)


# %%
def compute_sensitivity_by_cell_line(
    root,
    drug: str,
    cell_lines: "list[str]",
    response_feature: int,
    pooling: str = "max",
    dose_mode: str = "max",
    cell_agg: str = "mean_acts",
) -> "dict[str, float]":
    """
    Measure drug-sensitivity for each cell line.

    response_feature: typically the cytotox feature (cell death signature)

    Returns: {"SW48": 0.8, "LU65": 0.2, ...}
        High = drug kills cells (sensitive)
        Low = drug doesn't kill (resistant)
    """
    deltas = compute_drug_deltas_matrix(
        root, drug, cell_lines, pooling, dose_mode, cell_agg
    )
    # → (n_cell_lines, n_features)
    sensitivities = deltas[:, response_feature]  # extract one column
    return {cl: sensitivities[i].item() for i, cl in enumerate(cell_lines)}


def compute_control_features_by_cell_line(
    root, cell_lines: "list[str]", pooling: str = "max", cell_agg: str = "mean_acts"
) -> "torch.Tensor":
    """
    Get feature activations in CONTROL state (DMSO) for each cell line.
    Returns: (n_cell_lines, n_features)

    These are the PREDICTORS: which s0 features correlate with drug sensitivity?
    """
    raise NotImplementedError


def compute_feature_sensitivity_correlation(
    root,
    drug: str,
    cell_lines: "list[str]",
    response_feature: int,
    pooling: str = "max",
    dose_mode: str = "max",
    cell_agg: str = "mean_acts",
) -> "torch.Tensor":
    """
    Correlate each CONTROL feature with drug sensitivity across cell lines.

    Returns: (n_features,) with Pearson correlation per feature
    High positive = high s0 activation predicts high sensitivity
    """
    sensitivity = compute_sensitivity_by_cell_line(
        root, drug, cell_lines, response_feature, pooling, dose_mode, cell_agg
    )
    sens_vector = torch.tensor([sensitivity[cl] for cl in cell_lines])
    control_features = compute_control_features_by_cell_line(
        root, cell_lines, pooling, cell_agg
    )

    # Pearson correlation
    sens_centered = sens_vector - sens_vector.mean()
    feats_centered = control_features - control_features.mean(dim=0)
    cov = (feats_centered * sens_centered.unsqueeze(1)).sum(dim=0) / (
        len(cell_lines) - 1
    )
    correlations = cov / (sens_vector.std() * control_features.std(dim=0) + 1e-8)
    return correlations


# %% [markdown]
# ## Act 2 Expected Output
#
# ```
# Control features predicting ralimetinib sensitivity:
#   Feature 1234 (EGFR_signaling):  r = 0.67, p < 1e-7
#   Feature 5678 (RTK_activity):    r = 0.54, p < 1e-5
#
# NOT predictive (negative controls):
#   Feature 3456 (p38_stress):      r = 0.08, p = 0.58
#   Feature 7890 (p38_MAPK):        r = 0.04, p = 0.71
# ```
#
# **Negative Control Check:** p38 features are NOT predictive.
# If ralimetinib were actually a p38 inhibitor, p38 features should predict sensitivity.

# %% [markdown]
# ## Act 2 Extension: Mutation-Stratified Analysis
#
# Zoom in on EGFR-mutant cell lines specifically.
#
# **The question:** Which mutations confer sensitivity vs resistance?
#
# **Setup:**
# - EGFR-activating mutations (L858R, exon 19 del) → EGFR-dependent → SENSITIVE
# - EGFR resistance mutations (T790M) → less dependent → RESISTANT
#   - NOTE: We don't have T790M cell line, can't directly show resistance mutation
# - KRAS mutations → bypass mechanism → RESISTANT (negative control)
# - MAPK mutations → should have NO effect on sensitivity (negative control)
#
# **Analysis steps:**
# 1. Compare CONTROL feature activations across EGFR-mutant cell lines
#    - What are the major s0 differences between sensitive vs resistant cells?
#    - Look at upstream pathway features, not just EGFR itself
# 2. Correlate mutation status with drug sensitivity
# 3. Negative controls:
#    - KRAS mutations: confer RESISTANCE (bypass)
#    - MAPK mutations: NO effect


# %%
def compare_mutation_groups(
    root,
    drug: str,
    response_feature: int,
    mutation_key: str,  # e.g., "EGFR_mutation", "KRAS_mutation"
    pooling: str = "max",
    dose_mode: str = "max",
    cell_agg: str = "mean_acts",
):
    """
    Compare drug sensitivity between mutation-positive and mutation-negative cell lines.

    mutation_key: A metadata we construct from cell line annotations.
        Cell lines have known mutations (from DepMap, literature, etc.).
        We create a boolean metadata per mutation type, e.g.:
            root.metadatas["EGFR_mutation"] = is_egfr_mutant[cell_line_id]
            root.metadatas["KRAS_mutation"] = is_kras_mutant[cell_line_id]

    Returns: dict with sensitivity stats for each group
    """
    raise NotImplementedError


def s0_feature_differences_by_mutation(
    root,
    mutation_key: str,
    cell_lines: "list[str]",
    pooling: str = "max",
    cell_agg: str = "mean_acts",
    top_k: int = 20,
):
    """
    Find which CONTROL features differ most between mutation groups.

    Helps understand: what's different about these cells BEFORE drug treatment?

    mutation_key: see compare_mutation_groups - boolean metadata constructed from cell line annotations.
    """
    control_feats = compute_control_features_by_cell_line(
        root, cell_lines, pooling, cell_agg
    )
    # Split by mutation status, compare means, return top differing features
    raise NotImplementedError


# %% [markdown]
# ## Act 2 Alternative: Adapt Your find_predictive_features
#
# Your code does SAME-DOCUMENT prediction. We need CROSS-CONDITION prediction.
#
# **The bridge:** Compute drug-sensitivity from PERTURBED data, attach as metadata to CONTROL docs.
#
# **What is drug-sensitivity?**
# For each cell line, measure how much the CYTOTOX FEATURE increases when drug is applied:
# ```
# sensitivity[cell_line] = cytotox(s1_drug) - cytotox(s0_control)
# ```
# - High value (e.g., 0.8) = drug kills cells = SENSITIVE (like SW48)
# - Low value (e.g., 0.2) = drug doesn't kill = RESISTANT (like LU65, KRAS bypass)

# %%
# STEP 1: Compute drug-sensitivity per cell line
CYTOTOX_ID = 42  # the cytotox feature we identified in Act 1
cell_lines = get_all_cell_lines(root)
sensitivity = compute_sensitivity_by_cell_line(
    root, "ralimetinib", cell_lines, CYTOTOX_ID
)
# → {"SW48": 0.8, "LU65": 0.2, ...}  (cytotox delta per cell line)

# STEP 2: Binarize to sensitive/resistant
threshold = median(list(sensitivity.values()))
is_sensitive = {cl: 1 if s > threshold else 0 for cl, s in sensitivity.items()}
# → {"SW48": 1, "LU65": 0, ...}

# STEP 3: Create NEW metadata "sensitive" on control docs
# This is a join: each doc looks up its cell_line → is_sensitive[cell_line]
control_eval = root.open_filtered("dmso_only")
# ... create metadata where each doc's label = is_sensitive[doc.cell_line] ...
control_eval.metadatas["sensitive"] = ...  # the joined labels

# STEP 4: Use your find_predictive_features (unchanged!)
top_features, top_scores, scores = find_predictive_features(
    ev=control_eval,
    metadata_key="sensitive",
    target_value=1,  # find features enriched in sensitive cell lines
    top_k=20,
    pooling="max",
)

# %% [markdown]
# **Why this works:** Your code asks "which features fire more in sensitive vs resistant docs?"
# By attaching perturbed-derived labels to control docs, we ask:
# "which CONTROL features predict PERTURBED outcomes?"

# %% [markdown]
# ---
# # ACT 3: Mechanistic Attribution (PR #28)
#
# **Goal:** Show feature-feature attribution explaining mechanism.
#
# Compare SW48 (sensitive) vs LU65 (resistant):
# - SW48: EGFR → cytotox pathway dominates
# - LU65: KRAS → survival pathway dominates, bypasses EGFR
#
# **Depends on PR #28 (CLT/attribution graphs)**
#
# Punting detailed spec until PR #28 lands.

# %% [markdown]
# ---
# # Appendix: PRISM Validation (Adam handles this)
#
# Once `compute_drug_deltas_matrix` exists, we can validate the cytotox feature
# against external PRISM cell viability data.
#
# For a given drug, our cytotox feature activation profile across cell lines
# should match PRISM's killing profile for that drug.
#
# ```python
# # Get cytotox feature activation per cell line for a drug
# deltas = compute_drug_deltas_matrix(root, "doxorubicin", cell_lines)
# cytotox_activation = {cl: deltas[i, CYTOTOX_ID].item() for i, cl in enumerate(cell_lines)}
#
# # Load PRISM viability for same drug
# prism_viability = load_prism_data("doxorubicin")  # {cell_line: viability}
#
# # Correlate - expect NEGATIVE (high cytotox = low viability = more killing)
# # Expected: r ≈ -0.6 to -0.8
# # Repeat across multiple drugs to confirm feature generalizes
# ```
