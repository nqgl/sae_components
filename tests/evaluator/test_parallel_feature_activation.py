"""
Tests for the parallel feature activation pipeline.

These tests verify that the parallel implementation produces the same
results as the sequential implementation.
"""

import torch

try:
    import pytest
except ImportError:
    pytest = None

from saeco.evaluation.filtered import FilteredTensor
from saeco.evaluation.filtered_modular import Mask
from saeco.evaluation.return_objects import (
    AggregationType,
    Feature,
    SelectedDocs,
    TopActivations,
    _pk_to_k,
)


class MockEvaluation:
    """Mock Evaluation class for testing."""

    def __init__(self, num_docs: int = 100, seq_len: int = 32, d_dict: int = 10):
        self.num_docs = num_docs
        self.seq_len = seq_len
        self.d_dict = d_dict
        self.cuda = torch.device("cpu")
        self._features_data = {}

        # Generate synthetic feature activations
        for i in range(d_dict):
            # Each feature has sparse activations
            mask = torch.ones(num_docs, dtype=torch.bool)
            # Random activations with shape (num_docs, seq_len)
            values = torch.rand(num_docs, seq_len) * (i + 1)
            # Make it sparse by zeroing out random entries
            values = values * (torch.rand(num_docs, seq_len) > 0.7).float()
            self._features_data[i] = FilteredTensor.from_value_and_mask(
                value=values, mask_obj=mask
            )

    @property
    def features(self):
        return self

    def __getitem__(self, feature_id: int) -> FilteredTensor:
        return self._features_data[feature_id]


# -------------------------------------------------------------------------
# Test _pk_to_k helper
# -------------------------------------------------------------------------


def test_pk_to_k_with_k():
    """Test _pk_to_k when k is provided."""
    assert _pk_to_k(None, 10, 100) == 10
    assert _pk_to_k(None, 50, 100) == 50
    assert _pk_to_k(None, 150, 100) == 100  # Clamped to max


def test_pk_to_k_with_p():
    """Test _pk_to_k when p is provided."""
    assert _pk_to_k(0.1, None, 100) == 10
    assert _pk_to_k(0.5, None, 100) == 50
    assert _pk_to_k(1.0, None, 100) == 100


def test_pk_to_k_raises_on_both():
    """Test that providing both p and k raises an error."""
    try:
        _pk_to_k(0.1, 10, 100)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_pk_to_k_raises_on_neither():
    """Test that providing neither p nor k raises an error."""
    try:
        _pk_to_k(None, None, 100)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# -------------------------------------------------------------------------
# Test Feature class
# -------------------------------------------------------------------------


def test_feature_make():
    """Test Feature.make factory method."""
    mock_eval = MockEvaluation()

    # Make with feature_id
    f1 = Feature.make(src_eval=mock_eval, feature_id=0)
    assert f1.spec.feature_id == 0

    # Make with FilteredTensor
    ft = mock_eval.features[0]
    f2 = Feature.make(src_eval=mock_eval, feature=ft)
    assert f2.spec is ft


def test_feature_make_batch():
    """Test Feature.make_batch factory method."""
    mock_eval = MockEvaluation()

    features = Feature.make_batch(src_eval=mock_eval, features=[0, 1, 2])
    assert len(features) == 3
    for i, f in enumerate(features):
        assert f.spec.feature_id == i


def test_feature_aggregate():
    """Test Feature.aggregate method."""
    mock_eval = MockEvaluation()
    feature = Feature.make(src_eval=mock_eval, feature_id=0)

    # Test MAX aggregation
    agg_max = feature.aggregate(AggregationType.MAX)
    assert agg_max.value.shape == (mock_eval.num_docs,)

    # Test SUM aggregation
    agg_sum = feature.aggregate(AggregationType.SUM)
    assert agg_sum.value.shape == (mock_eval.num_docs,)

    # Test MEAN aggregation
    agg_mean = feature.aggregate(AggregationType.MEAN)
    assert agg_mean.value.shape == (mock_eval.num_docs,)


def test_feature_top_activations():
    """Test Feature.top_activations method."""
    mock_eval = MockEvaluation()
    feature = Feature.make(src_eval=mock_eval, feature_id=0)

    top_acts = feature.top_activations(k=10)

    assert isinstance(top_acts, TopActivations)
    assert top_acts.doc_selection.doc_indices.shape == (10,)


# -------------------------------------------------------------------------
# Test batched operations
# -------------------------------------------------------------------------


def test_batched_top_activations():
    """Test Feature.batched_top_activations produces correct results."""
    mock_eval = MockEvaluation()

    # Create multiple features
    features = Feature.make_batch(src_eval=mock_eval, features=[0, 1, 2, 3])

    # Get batched results
    batched_results = Feature.batched_top_activations(features, k=10)

    assert len(batched_results) == 4

    # Verify each result
    for i, result in enumerate(batched_results):
        assert isinstance(result, TopActivations)
        assert result.doc_selection.doc_indices.shape == (10,)

        # Compare with sequential result
        sequential_result = features[i].top_activations(k=10)

        # The doc_indices should be the same
        assert torch.equal(
            result.doc_selection.doc_indices, sequential_result.doc_selection.doc_indices
        )


def test_batched_top_activations_empty():
    """Test batched_top_activations with empty list."""
    results = Feature.batched_top_activations([], k=10)
    assert results == []


def test_batched_top_activations_different_aggregations():
    """Test batched_top_activations with different aggregation types."""
    mock_eval = MockEvaluation()
    features = Feature.make_batch(src_eval=mock_eval, features=[0, 1])

    for agg in [AggregationType.MAX, AggregationType.SUM, AggregationType.MEAN]:
        batched = Feature.batched_top_activations(features, agg=agg, k=5)
        sequential = [f.top_activations(agg=agg, k=5) for f in features]

        for b, s in zip(batched, sequential):
            assert torch.equal(b.doc_selection.doc_indices, s.doc_selection.doc_indices)


def test_batched_consistency_with_sequential():
    """Test that batched and sequential produce identical results."""
    mock_eval = MockEvaluation(num_docs=50, d_dict=5)

    feature_ids = [0, 1, 2, 3, 4]
    features = Feature.make_batch(src_eval=mock_eval, features=feature_ids)

    # Get results using batched method
    batched_results = Feature.batched_top_activations(features, k=10)

    # Get results using sequential method
    sequential_results = [f.top_activations(k=10) for f in features]

    # Compare
    for i, (batched, sequential) in enumerate(
        zip(batched_results, sequential_results)
    ):
        assert torch.equal(
            batched.doc_selection.doc_indices, sequential.doc_selection.doc_indices
        ), f"Mismatch for feature {i}"


# -------------------------------------------------------------------------
# Test with proportion (p) instead of k
# -------------------------------------------------------------------------


def test_batched_top_activations_with_p():
    """Test batched_top_activations with proportion p."""
    mock_eval = MockEvaluation(num_docs=100)
    features = Feature.make_batch(src_eval=mock_eval, features=[0, 1])

    batched = Feature.batched_top_activations(features, p=0.1)

    # p=0.1 with 100 docs should give 10 results
    for result in batched:
        assert result.doc_selection.doc_indices.shape == (10,)


# -------------------------------------------------------------------------
# Test TopActivations properties
# -------------------------------------------------------------------------


def test_top_activations_acts_property():
    """Test TopActivations.acts property."""
    mock_eval = MockEvaluation()
    feature = Feature.make(src_eval=mock_eval, feature_id=0)
    top_acts = feature.top_activations(k=10)

    # Acts should be the feature activations for selected docs
    acts = top_acts.acts
    assert acts.shape[0] == 10


def test_top_activations_docs_property():
    """Test TopActivations.docs property."""
    mock_eval = MockEvaluation()

    # Add mock docs to the mock evaluation
    mock_eval.docs = torch.randint(0, 1000, (mock_eval.num_docs, mock_eval.seq_len))

    feature = Feature.make(src_eval=mock_eval, feature_id=0)
    top_acts = feature.top_activations(k=10)

    # Docs should be accessible via doc_selection
    docs = top_acts.docs
    assert docs.shape == (10, mock_eval.seq_len)


if __name__ == "__main__":
    # Run tests
    test_pk_to_k_with_k()
    test_pk_to_k_with_p()
    test_feature_make()
    test_feature_make_batch()
    test_feature_aggregate()
    test_feature_top_activations()
    test_batched_top_activations()
    test_batched_top_activations_empty()
    test_batched_top_activations_different_aggregations()
    test_batched_consistency_with_sequential()
    test_batched_top_activations_with_p()
    print("All tests passed!")
