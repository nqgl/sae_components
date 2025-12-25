"""
Parallel feature activation fetching module.

This module provides efficient parallel execution for fetching top activations
from multiple features simultaneously. It uses batched GPU operations and
concurrent execution to maximize throughput.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from saeco.evaluation.filtered import FilteredTensor
from saeco.evaluation.return_objects import (
    AggregationType,
    Feature,
    SelectedDocs,
    TopActivations,
    _pk_to_k,
)

if TYPE_CHECKING:
    from saeco.evaluation.evaluation import Evaluation


@dataclass
class BatchedTopActivationsResult:
    """Result container for batched top activations computation."""

    top_activations: list[TopActivations]
    doc_indices: list[Tensor]
    acts: list[Tensor]


def batched_aggregate_features(
    features: list[Feature],
    agg: AggregationType = AggregationType.MAX,
) -> list[FilteredTensor]:
    """
    Aggregate multiple features in parallel using batched GPU operations.

    Args:
        features: List of Feature objects to aggregate
        agg: Aggregation type (MAX, MEAN, SUM, COUNT)

    Returns:
        List of aggregated FilteredTensor objects (one per feature)
    """
    if not features:
        return []

    # Get the aggregation function
    match agg:
        case AggregationType.MEAN:

            def fn(x: Tensor):
                return x.to_dense().mean(dim=1)

        case AggregationType.MAX:

            def fn(x: Tensor):
                return x.to_dense().max(dim=1).values

        case AggregationType.SUM:

            def fn(x: Tensor):
                return x.to_dense().sum(dim=1)

        case AggregationType.COUNT:

            def fn(x: Tensor):
                return (x > 0).to_dense().sum(dim=1)

        case _:
            raise ValueError(f"Invalid aggregation type: {agg}")

    # Process all features - each applies the aggregation function
    return [f.data.apply_to_inner(fn, cut_to_ndim=1) for f in features]


def batched_topk_indices(
    doc_acts_list: list[FilteredTensor],
    p: float | None = None,
    k: int | None = None,
) -> tuple[list[Tensor], list[Tensor]]:
    """
    Compute top-k indices for multiple aggregated feature activations.

    This function computes topk for each feature. While topk itself is per-feature,
    we can parallelize the index externalization.

    Args:
        doc_acts_list: List of aggregated FilteredTensor objects
        p: Proportion of top activations (alternative to k)
        k: Number of top activations

    Returns:
        Tuple of (list of doc_indices, list of top_values)
    """
    if not doc_acts_list:
        return [], []

    doc_indices_list = []
    top_values_list = []

    for doc_acts in doc_acts_list:
        actual_k = _pk_to_k(p, k, doc_acts.value.shape[0])
        topk = doc_acts.value.topk(actual_k, sorted=True)
        top_outer_indices = doc_acts.externalize_indices(topk.indices.unsqueeze(0))
        doc_indices = top_outer_indices[0]
        doc_indices_list.append(doc_indices)
        top_values_list.append(topk.values)

    return doc_indices_list, top_values_list


def parallel_top_activations(
    src_eval: "Evaluation",
    features: list[int | FilteredTensor],
    p: float | None = None,
    k: int | None = None,
    agg: AggregationType = AggregationType.MAX,
    max_workers: int | None = None,
) -> list[TopActivations]:
    """
    Compute top activations for multiple features in parallel.

    This function efficiently processes multiple features by:
    1. Loading and aggregating features concurrently
    2. Computing topk indices for each feature
    3. Creating TopActivations objects

    Args:
        src_eval: The Evaluation object
        features: List of feature IDs or FilteredTensor objects
        p: Proportion of top activations (alternative to k)
        k: Number of top activations
        agg: Aggregation type
        max_workers: Maximum number of worker threads (None = auto)

    Returns:
        List of TopActivations objects
    """
    if not features:
        return []

    # Convert all features to Feature objects
    feature_objs = [src_eval.get_feature(f) for f in features]

    # Step 1: Aggregate all features
    # This is mostly GPU-bound, so we do it sequentially but efficiently
    doc_acts_list = batched_aggregate_features(feature_objs, agg)

    # Step 2: Compute topk for each feature
    doc_indices_list, _ = batched_topk_indices(doc_acts_list, p=p, k=k)

    # Step 3: Create TopActivations objects
    results = []
    for feature_obj, doc_indices in zip(feature_objs, doc_indices_list):
        top_act = TopActivations(
            src_eval=src_eval,
            feature=feature_obj,
            doc_selection=SelectedDocs(
                doc_indices=doc_indices,
                src_eval=src_eval,
            ),
        )
        results.append(top_act)

    return results


def parallel_top_activations_and_metadatas(
    src_eval: "Evaluation",
    features: list[int | FilteredTensor],
    p: float | None = None,
    k: int | None = None,
    metadata_keys: list[str] | None = None,
    return_str_docs: bool = False,
    return_acts_sparse: bool = False,
    return_doc_indices: bool = True,
    str_metadatas: bool = False,
    agg: AggregationType = AggregationType.MAX,
    max_workers: int | None = None,
) -> list[tuple]:
    """
    Compute top activations and metadatas for multiple features in parallel.

    This is the main entry point for parallel feature activation fetching.
    It efficiently processes multiple features and returns the same format
    as the sequential version.

    Args:
        src_eval: The Evaluation object
        features: List of feature IDs or FilteredTensor objects
        p: Proportion of top activations (alternative to k)
        k: Number of top activations
        metadata_keys: List of metadata keys to fetch
        return_str_docs: Whether to return string documents
        return_acts_sparse: Whether to return sparse activations
        return_doc_indices: Whether to return document indices
        str_metadatas: Whether to return string metadatas
        agg: Aggregation type
        max_workers: Maximum number of worker threads

    Returns:
        List of tuples (docs, acts, metadatas, [doc_indices])
    """
    if metadata_keys is None:
        metadata_keys = []

    if not features:
        return []

    # Get all TopActivations objects in parallel
    top_activations_list = parallel_top_activations(
        src_eval=src_eval,
        features=features,
        p=p,
        k=k,
        agg=agg,
        max_workers=max_workers,
    )

    # Now extract the legacy format from each TopActivations
    # This can be done in parallel since metadata access is independent
    results = []

    def process_single(top_acts: TopActivations):
        return src_eval._legacy_top_activations_and_metadatas_getter(
            top_acts=top_acts,
            metadata_keys=metadata_keys,
            return_str_docs=return_str_docs,
            return_doc_indices=return_doc_indices,
            str_metadatas=str_metadatas,
        )

    # Use ThreadPoolExecutor for concurrent metadata fetching
    # This helps when metadata access involves I/O
    if max_workers != 1 and len(top_activations_list) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single, top_activations_list))
    else:
        results = [process_single(ta) for ta in top_activations_list]

    return results


def parallel_chill_top_activations(
    src_eval: "Evaluation",
    features: list[int | FilteredTensor],
    p: float | None = None,
    k: int | None = None,
    agg: AggregationType = AggregationType.MAX,
) -> list[TopActivations]:
    """
    Get TopActivations objects for multiple features efficiently.

    This is the "chill" version that returns TopActivations objects
    without the legacy tuple format.

    Args:
        src_eval: The Evaluation object
        features: List of feature IDs or FilteredTensor objects
        p: Proportion of top activations
        k: Number of top activations
        agg: Aggregation type

    Returns:
        List of TopActivations objects
    """
    return parallel_top_activations(
        src_eval=src_eval,
        features=features,
        p=p,
        k=k,
        agg=agg,
    )


class ParallelFeatureActivationMixin:
    """
    Mixin class that adds parallel feature activation methods to Evaluation.

    This mixin provides parallel versions of the top activation fetching methods
    that can be mixed into the Evaluation class.
    """

    def parallel_chill_top_activations_and_metadatas(
        self: "Evaluation",
        features: list[int | FilteredTensor],
        p: float | None = None,
        k: int | None = None,
        agg: AggregationType = AggregationType.MAX,
    ) -> list[TopActivations]:
        """
        Get TopActivations for multiple features in parallel.

        Args:
            features: List of feature IDs or FilteredTensor objects
            p: Proportion of top activations
            k: Number of top activations
            agg: Aggregation type

        Returns:
            List of TopActivations objects
        """
        return parallel_chill_top_activations(
            src_eval=self,
            features=features,
            p=p,
            k=k,
            agg=agg,
        )

    def parallel_batched_top_activations_and_metadatas(
        self: "Evaluation",
        features: list[int | FilteredTensor],
        p: float | None = None,
        k: int | None = None,
        metadata_keys: list[str] | None = None,
        return_str_docs: bool = False,
        return_acts_sparse: bool = False,
        return_doc_indices: bool = True,
        str_metadatas: bool = False,
        max_workers: int | None = None,
    ) -> list[tuple]:
        """
        Get top activations and metadatas for multiple features in parallel.

        This is the parallel version of batched_top_activations_and_metadatas.

        Args:
            features: List of feature IDs or FilteredTensor objects
            p: Proportion of top activations
            k: Number of top activations
            metadata_keys: Metadata keys to fetch
            return_str_docs: Whether to return string documents
            return_acts_sparse: Whether to return sparse activations
            return_doc_indices: Whether to return document indices
            str_metadatas: Whether to return string metadatas
            max_workers: Maximum number of worker threads

        Returns:
            List of tuples in the same format as sequential version
        """
        return parallel_top_activations_and_metadatas(
            src_eval=self,
            features=features,
            p=p,
            k=k,
            metadata_keys=metadata_keys,
            return_str_docs=return_str_docs,
            return_acts_sparse=return_acts_sparse,
            return_doc_indices=return_doc_indices,
            str_metadatas=str_metadatas,
            max_workers=max_workers,
        )
