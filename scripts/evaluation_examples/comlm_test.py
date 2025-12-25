# %%
import torch
from comlm.datasource.training_batch import NoisedBatch
from load_comlm import root_eval

# %%
from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig
from saeco.data.config.model_config.model_cfg import ModelConfig
from saeco.evaluation.evaluation import Evaluation

# %%

root: Evaluation[NoisedBatch] = root_eval
# %%
# ADD METADATA

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


data0 = root.docs[0:4]
# %%

root.get_inputs_type()
# %%
root.saved_acts.get_inputs_type()
# %%
#
# Initialize metadata if necessary
#
metadata_keys = list(metadata_tokenizer.tokenizers.keys())
has_metadata = [key in root.metadatas for key in metadata_keys]
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
        root.metadatas[key] = metadata
        root.metadatas.set_str_translator(
            key, {"<<PAD>>": 0, "<<UNK>>": 1, **tokenizer.tokens}
        )
# %%
acts = root.chill_top_activations_and_metadatas(
    7,
    k=4,
    # metadata_keys=metadata_keys,
)


# %%
acts.doc_selection.metadata[metadata_keys[0]]
# %%
root.top_activations_metadata_enrichments(feature=7, metadata_keys=metadata_keys, p=0.1)
# %%
ac = root.activation_cosims(out_device="cpu", blocks_per_dim=2)

# %%
root.d_dict**2 / 1e9

# %%
# =============================================================================
# FEATURE METADATA PREDICTION ANALYSIS
# =============================================================================
# Find features that predict a particular value of a particular metadata category
# This is the higher-level abstract form of the task (e.g., find features that predict cancer)

import tqdm
from dataclasses import dataclass
from typing import Optional
import einops


@dataclass
class FeatureMetadataScores:
    """Stores enrichment scores for all features for a given metadata key and value."""

    metadata_key: str
    target_value: int  # The token ID for the target value
    target_value_str: str  # Human-readable name
    feature_scores: torch.Tensor  # Shape: (d_dict,) - enrichment score per feature
    feature_counts_in_target: torch.Tensor  # Shape: (d_dict,) - how many target docs each feature fires on
    feature_counts_total: torch.Tensor  # Shape: (d_dict,) - total docs each feature fires on
    target_count: int  # Total docs with target value
    total_docs: int  # Total docs in dataset


def compute_feature_activity_per_doc(
    ev: Evaluation, pooling: str = "any", device: str = "cuda"
) -> torch.Tensor:
    """
    Compute per-document feature activity.

    Args:
        ev: Evaluation object
        pooling: How to aggregate across sequence positions:
            - "any": 1 if feature fires anywhere in doc, 0 otherwise
            - "max": max activation in doc
            - "mean": mean activation across doc
            - "sum": sum of activations in doc

    Returns:
        Tensor of shape (num_docs, d_dict) with per-document feature activity
    """
    result = torch.zeros(ev.num_docs, ev.d_dict, device=device)
    doc_offset = 0

    for chunk in tqdm.tqdm(ev.saved_acts.chunks, desc="Computing feature activity"):
        acts = chunk.acts.value.to(device).to_dense()  # (docs_in_chunk, seq_len, d_dict)
        num_docs_chunk = acts.shape[0]

        if pooling == "any":
            chunk_result = (acts > 0).any(dim=1).float()
        elif pooling == "max":
            chunk_result = acts.max(dim=1).values
        elif pooling == "mean":
            chunk_result = acts.mean(dim=1)
        elif pooling == "sum":
            chunk_result = acts.sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        result[doc_offset : doc_offset + num_docs_chunk] = chunk_result
        doc_offset += num_docs_chunk

    return result


def compute_feature_metadata_scores(
    ev: Evaluation,
    metadata_key: str,
    target_value: int | str,
    feature_activity: Optional[torch.Tensor] = None,
    pooling: str = "any",
    device: str = "cuda",
) -> FeatureMetadataScores:
    """
    Compute enrichment scores for all features predicting a specific metadata value.

    Args:
        ev: Evaluation object
        metadata_key: Name of the metadata field (e.g., "cancer_type", "tissue")
        target_value: The target value to predict (int token ID or str label)
        feature_activity: Pre-computed feature activity matrix (optional, will compute if not provided)
        pooling: How to aggregate feature activity across sequence positions
        device: Device to compute on

    Returns:
        FeatureMetadataScores containing scores and counts for all features
    """
    # Convert string target to int if needed
    if isinstance(target_value, str):
        meta_info = ev.metadatas.get(metadata_key)
        target_value_int = meta_info.info.fromstr[target_value]
        target_value_str = target_value
    else:
        target_value_int = target_value
        target_value_str = ev.metadatas.get(metadata_key).info.tostr.get(
            target_value, str(target_value)
        )

    # Get metadata for all docs
    metadata = ev._root_metadatas[metadata_key].to(device)

    # Compute feature activity if not provided
    if feature_activity is None:
        feature_activity = compute_feature_activity_per_doc(ev, pooling=pooling, device=device)

    # Create mask for target documents
    target_mask = (metadata == target_value_int).float()  # (num_docs,)
    target_count = int(target_mask.sum().item())

    # Compute counts: how many docs with target value does each feature fire on
    # feature_activity: (num_docs, d_dict), target_mask: (num_docs,)
    feature_counts_in_target = (feature_activity * target_mask.unsqueeze(1)).sum(dim=0)  # (d_dict,)

    # Total docs each feature fires on
    feature_counts_total = feature_activity.sum(dim=0)  # (d_dict,)

    # Compute enrichment score using log-odds ratio with smoothing
    # P(feature fires | target) vs P(feature fires | not target)
    eps = 1e-6
    non_target_count = ev.num_docs - target_count + eps
    feature_counts_not_target = feature_counts_total - feature_counts_in_target

    # Rates
    rate_in_target = (feature_counts_in_target + eps) / (target_count + eps)
    rate_not_target = (feature_counts_not_target + eps) / non_target_count

    # Log odds ratio as enrichment score
    feature_scores = torch.log2(rate_in_target / (rate_not_target + eps))

    return FeatureMetadataScores(
        metadata_key=metadata_key,
        target_value=target_value_int,
        target_value_str=target_value_str,
        feature_scores=feature_scores.cpu(),
        feature_counts_in_target=feature_counts_in_target.cpu(),
        feature_counts_total=feature_counts_total.cpu(),
        target_count=target_count,
        total_docs=ev.num_docs,
    )


def find_predictive_features(
    ev: Evaluation,
    metadata_key: str,
    target_value: int | str,
    top_k: int = 20,
    min_target_count: int = 5,
    feature_activity: Optional[torch.Tensor] = None,
    pooling: str = "any",
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, FeatureMetadataScores]:
    """
    Find features most predictive of a metadata value.

    Args:
        ev: Evaluation object
        metadata_key: Name of the metadata field
        target_value: The target value to predict
        top_k: Number of top features to return
        min_target_count: Minimum number of target docs a feature must fire on
        feature_activity: Pre-computed feature activity matrix (optional)
        pooling: How to aggregate feature activity across sequence positions
        device: Device to compute on

    Returns:
        Tuple of (feature_ids, scores, full_scores_object)
    """
    scores = compute_feature_metadata_scores(
        ev=ev,
        metadata_key=metadata_key,
        target_value=target_value,
        feature_activity=feature_activity,
        pooling=pooling,
        device=device,
    )

    # Filter by minimum count in target
    valid_mask = scores.feature_counts_in_target >= min_target_count
    masked_scores = scores.feature_scores.clone()
    masked_scores[~valid_mask] = float("-inf")

    # Get top-k
    top_scores, top_indices = masked_scores.topk(top_k)

    return top_indices, top_scores, scores


def print_predictive_features(
    ev: Evaluation,
    metadata_key: str,
    target_value: int | str,
    top_k: int = 20,
    min_target_count: int = 5,
    show_examples: int = 3,
    feature_activity: Optional[torch.Tensor] = None,
):
    """
    Find and print features most predictive of a metadata value, with examples.

    Args:
        ev: Evaluation object
        metadata_key: Name of the metadata field
        target_value: The target value to predict
        top_k: Number of top features to show
        min_target_count: Minimum number of target docs a feature must fire on
        show_examples: Number of example documents to show per feature
        feature_activity: Pre-computed feature activity matrix (optional)
    """
    top_indices, top_scores, scores = find_predictive_features(
        ev=ev,
        metadata_key=metadata_key,
        target_value=target_value,
        top_k=top_k,
        min_target_count=min_target_count,
        feature_activity=feature_activity,
    )

    print(f"\n{'=' * 60}")
    print(f"Top {top_k} features predicting {metadata_key}={scores.target_value_str}")
    print(f"Target docs: {scores.target_count}/{scores.total_docs} ({100*scores.target_count/scores.total_docs:.2f}%)")
    print(f"{'=' * 60}\n")

    for rank, (feat_id, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist())):
        if score == float("-inf"):
            break

        count_target = int(scores.feature_counts_in_target[feat_id].item())
        count_total = int(scores.feature_counts_total[feat_id].item())
        precision = count_target / max(count_total, 1)

        print(f"Rank {rank + 1}: Feature {feat_id}")
        print(f"  Score: {score:.3f} (log2 odds ratio)")
        print(f"  Fires on: {count_target}/{scores.target_count} target docs ({100*count_target/scores.target_count:.1f}% recall)")
        print(f"  Precision: {count_target}/{count_total} total docs ({100*precision:.1f}%)")

        if show_examples > 0:
            # Show some example activations for this feature
            try:
                top_acts = ev.chill_top_activations_and_metadatas(feat_id, k=show_examples)
                mds = top_acts.doc_selection.metadata[metadata_key]
                print(f"  Example doc metadatas: {mds.as_str[:show_examples]}")
            except Exception as e:
                print(f"  (Could not get examples: {e})")
        print()


def compute_all_metadata_predictors(
    ev: Evaluation,
    metadata_key: str,
    top_k_per_value: int = 10,
    min_target_count: int = 5,
    min_value_count: int = 10,
    pooling: str = "any",
    device: str = "cuda",
) -> dict[str, tuple[torch.Tensor, torch.Tensor, FeatureMetadataScores]]:
    """
    Find top predictive features for each value of a metadata field.

    Args:
        ev: Evaluation object
        metadata_key: Name of the metadata field
        top_k_per_value: Number of top features per value
        min_target_count: Minimum feature count in target
        min_value_count: Minimum number of docs for a value to be included
        pooling: How to aggregate feature activity
        device: Device to compute on

    Returns:
        Dict mapping value_str -> (feature_ids, scores, full_scores)
    """
    # Pre-compute feature activity once
    feature_activity = compute_feature_activity_per_doc(ev, pooling=pooling, device=device)

    # Get all unique values for this metadata
    label_counts = ev.cached_call._metadata_unique_labels_and_counts_tensor(metadata_key)

    results = {}
    for label, count in zip(label_counts.labels.tolist(), label_counts.counts.tolist()):
        if count < min_value_count:
            continue

        label_str = ev.metadatas.get(metadata_key).info.tostr.get(label, str(label))

        top_indices, top_scores, scores = find_predictive_features(
            ev=ev,
            metadata_key=metadata_key,
            target_value=label,
            top_k=top_k_per_value,
            min_target_count=min_target_count,
            feature_activity=feature_activity,
            device=device,
        )

        results[label_str] = (top_indices, top_scores, scores)

    return results


# %%
# =============================================================================
# EXAMPLE USAGE: Finding metadata-predictive features
# =============================================================================
# First, let's see what metadata keys are available


def show_available_metadata(ev: Evaluation, md_keys: list[str]):
    """Show available metadata keys and their values."""
    print("Available metadata keys:")
    for key in md_keys:
        meta = ev.metadatas.get(key)
        label_counts = ev.cached_call._metadata_unique_labels_and_counts_tensor(key)
        print(f"  {key}: {len(label_counts.labels)} unique values")
        # Show top 5 values
        sorted_idx = label_counts.counts.argsort(descending=True)[:5]
        for idx in sorted_idx:
            label = label_counts.labels[idx].item()
            count = label_counts.counts[idx].item()
            label_str = meta.info.tostr.get(label, str(label))
            print(f"    {label_str}: {count} docs")


show_available_metadata(root, metadata_keys)

# %%
# =============================================================================
# Find features that predict a specific metadata value
# =============================================================================
# Example: Find features predictive of a particular cancer type or tissue
# Replace the metadata_key and target_value with actual values from your dataset

# Pre-compute feature activity once (can be reused for multiple queries)
print("Pre-computing feature activity per document...")
feature_activity = compute_feature_activity_per_doc(root, pooling="any", device="cuda")

# %%
# Example: Find features for the first metadata key and its most common value
example_metadata_key = metadata_keys[0]
example_label_counts = root.cached_call._metadata_unique_labels_and_counts_tensor(example_metadata_key)
most_common_value = example_label_counts.labels[example_label_counts.counts.argmax()].item()
most_common_value_str = root.metadatas.get(example_metadata_key).info.tostr.get(
    most_common_value, str(most_common_value)
)

print(f"\nExample: Finding features predictive of {example_metadata_key}='{most_common_value_str}'")

# Find top predictive features
top_features, top_scores, scores = find_predictive_features(
    ev=root,
    metadata_key=example_metadata_key,
    target_value=most_common_value,
    top_k=10,
    min_target_count=3,
    feature_activity=feature_activity,
)

# Print results
print_predictive_features(
    ev=root,
    metadata_key=example_metadata_key,
    target_value=most_common_value,
    top_k=10,
    min_target_count=3,
    show_examples=2,
    feature_activity=feature_activity,
)

# %%
# =============================================================================
# Find predictive features for ALL values of a metadata field
# =============================================================================


def summarize_all_predictors(
    ev: Evaluation,
    metadata_key: str,
    top_k_per_value: int = 5,
    min_target_count: int = 3,
    min_value_count: int = 10,
    feature_activity: Optional[torch.Tensor] = None,
):
    """Summarize top predictive features for each value of a metadata field."""
    results = compute_all_metadata_predictors(
        ev=ev,
        metadata_key=metadata_key,
        top_k_per_value=top_k_per_value,
        min_target_count=min_target_count,
        min_value_count=min_value_count,
        feature_activity=feature_activity if feature_activity is not None else None,
    )

    print(f"\n{'=' * 70}")
    print(f"Summary: Top {top_k_per_value} features for each {metadata_key} value")
    print(f"{'=' * 70}")

    for value_str, (feat_ids, feat_scores, full_scores) in sorted(results.items()):
        print(f"\n{value_str} ({full_scores.target_count} docs):")
        for i, (fid, score) in enumerate(zip(feat_ids[:3].tolist(), feat_scores[:3].tolist())):
            if score == float("-inf"):
                break
            count_in = int(full_scores.feature_counts_in_target[fid].item())
            count_tot = int(full_scores.feature_counts_total[fid].item())
            print(f"  #{i+1} Feature {fid}: score={score:.2f}, {count_in}/{count_tot} precision")

    return results


# Example: Summarize predictors for the first metadata key
print(f"\nSummarizing predictors for {example_metadata_key}...")
all_predictors = summarize_all_predictors(
    ev=root,
    metadata_key=example_metadata_key,
    top_k_per_value=5,
    min_target_count=3,
    min_value_count=10,
    feature_activity=feature_activity,
)


# %%
# =============================================================================
# CANCER-SPECIFIC EXAMPLE (if cancer metadata exists)
# =============================================================================
# Look for cancer-related metadata keys


def find_cancer_predictive_features(ev: Evaluation, md_keys: list[str], feature_activity: torch.Tensor):
    """Find features predictive of cancer-related metadata values."""
    cancer_keywords = ["cancer", "tumor", "carcinoma", "leukemia", "lymphoma", "sarcoma", "melanoma"]

    for key in md_keys:
        meta = ev.metadatas.get(key)
        label_counts = ev.cached_call._metadata_unique_labels_and_counts_tensor(key)

        for label_idx, (label, count) in enumerate(
            zip(label_counts.labels.tolist(), label_counts.counts.tolist())
        ):
            if count < 10:  # Skip rare values
                continue

            label_str = meta.info.tostr.get(label, str(label))

            # Check if this looks cancer-related
            if any(kw in label_str.lower() for kw in cancer_keywords):
                print(f"\n{'=' * 60}")
                print(f"CANCER-RELATED: {key}='{label_str}' ({count} docs)")
                print(f"{'=' * 60}")

                print_predictive_features(
                    ev=ev,
                    metadata_key=key,
                    target_value=label,
                    top_k=10,
                    min_target_count=3,
                    show_examples=2,
                    feature_activity=feature_activity,
                )


# Run cancer feature search
print("\nSearching for cancer-related metadata values and their predictive features...")
find_cancer_predictive_features(root, metadata_keys, feature_activity)

# %%
