import pytest
import torch
from torch import Tensor

# Adjust this import to match your actual module path
from saeco.evaluation.filtered import Filter, FilteredTensor


def make_mask(N: int, true_idx: tuple[int, ...]) -> Tensor:
    mask = torch.zeros(N, dtype=torch.bool)
    mask[list(true_idx)] = True
    return mask


def make_filtered_dense(
    n_docs: int = 5,
    n_features: int = 3,
    true_idx: tuple[int, ...] = (0, 2, 4),
) -> tuple[FilteredTensor, Tensor, Tensor]:
    """
    Utility: build a dense FilteredTensor where the outer shape is (n_docs, n_features),
    outer mask selects `true_idx`, and inner value is compressed to mask.sum().
    """
    base = torch.arange(n_docs * n_features, dtype=torch.float32).reshape(
        n_docs, n_features
    )
    mask = make_mask(n_docs, true_idx)
    # from_unmasked_value expects `value` to be *unmasked* and will apply the mask
    ft = FilteredTensor.from_unmasked_value(value=base, filter_obj=mask)
    assert ft.shape == base.shape
    assert ft.value.shape[0] == mask.sum()
    return ft, base, mask


def make_filtered_sparse() -> tuple[FilteredTensor, Tensor]:
    """
    Utility: build a sparse FilteredTensor whose unfiltered dense representation
    is easy to reason about.

    Outer dense:
        doc 0: [0, 1, 0, 0]
        doc 1: [2, 0, 0, 0]
        doc 2: [0, 0, 3, 0]
        doc 3: [4, 0, 5, 0]
        doc 4: [0, 0, 0, 6]

    Mask keeps docs 1, 2, and 4 -> 3 active docs.
    """
    outer_dense = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],  # doc 0
            [2.0, 0.0, 0.0, 0.0],  # doc 1
            [0.0, 0.0, 3.0, 0.0],  # doc 2
            [4.0, 0.0, 5.0, 0.0],  # doc 3
            [0.0, 0.0, 0.0, 6.0],  # doc 4
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([False, True, True, False, True], dtype=torch.bool)

    inner_dense = outer_dense[mask]  # (3, 4)
    inner_sparse = inner_dense.to_sparse_coo()

    # Here `value` is already masked/compacted; `mask_obj` is the outer mask.
    ft = FilteredTensor.from_value_and_mask(
        value=inner_sparse,
        mask_obj=mask,
    )
    assert ft.is_sparse
    assert ft.shape == outer_dense.shape
    assert ft.value.shape == inner_dense.shape
    return ft, outer_dense


def make_filtered_sparse_with_mask() -> tuple[FilteredTensor, Tensor, Tensor]:
    """
    Utility: build a sparse FilteredTensor whose unfiltered dense representation
    is easy to reason about.

    Outer dense:
        doc 0: [0, 1, 0, 0]
        doc 1: [2, 0, 0, 0]
        doc 2: [0, 0, 3, 0]
        doc 3: [4, 0, 5, 0]
        doc 4: [0, 0, 0, 6]

    Mask keeps docs 1, 2, and 4 -> 3 active docs.
    """
    outer_dense = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],  # doc 0
            [2.0, 0.0, 0.0, 0.0],  # doc 1
            [0.0, 0.0, 3.0, 0.0],  # doc 2
            [4.0, 0.0, 5.0, 0.0],  # doc 3
            [0.0, 0.0, 0.0, 6.0],  # doc 4
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([False, True, True, False, True], dtype=torch.bool)

    inner_dense = outer_dense[mask]  # (3, 4)
    inner_sparse = inner_dense.to_sparse_coo()

    # Here `value` is already masked/compacted; `mask_obj` is the outer mask.
    ft = FilteredTensor.from_value_and_mask(
        value=inner_sparse,
        mask_obj=mask,
    )
    assert ft.is_sparse
    assert ft.shape == outer_dense.shape
    assert ft.value.shape == inner_dense.shape
    return ft, outer_dense, mask


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_from_value_and_mask_unmasked_identity():
    v = torch.randn(4, 3)
    ft = FilteredTensor.from_value_and_mask(v, mask_obj=None)

    # outer shape should be the same as the input tensor
    assert ft.shape == v.shape

    # mask should be along the first (doc) dimension and all True
    assert ft.filter.mask.shape == (v.shape[0],)
    assert ft.filter.mask.dtype == torch.bool
    assert ft.filter.mask.all()

    # value should be unchanged in shape and contents
    assert ft.value.shape == v.shape
    assert torch.allclose(ft.value, v)


def test_from_value_and_mask_with_compact_value_and_outer_mask():
    base = torch.arange(10.0).reshape(5, 2)
    outer_mask = torch.tensor([True, False, True, False, True], dtype=torch.bool)

    compact_value = base[outer_mask]

    ft = FilteredTensor.from_value_and_mask(value=compact_value, mask_obj=outer_mask)

    assert ft.shape == base.shape  # outer shape
    assert ft.value.shape == compact_value.shape
    assert torch.allclose(ft.value, compact_value)
    assert torch.equal(ft.filter.mask, outer_mask)


def test_from_unmasked_value_with_tensor_mask_applies_mask():
    base = torch.arange(20.0).reshape(5, 4)
    mask = torch.tensor([True, False, True, False, True], dtype=torch.bool)

    ft = FilteredTensor.from_unmasked_value(value=base, filter_obj=mask)

    # outer shape should be the original base shape
    assert ft.shape == base.shape

    # inner shape should be (#True, feature_dims)
    expected_inner = base[mask]
    assert ft.value.shape == expected_inner.shape
    assert torch.allclose(ft.value, expected_inner)

    # filter mask should be exactly the provided mask
    assert torch.equal(ft.filter.mask, mask)


def test_from_unmasked_value_with_filter_object():
    base = torch.arange(12.0).reshape(4, 3)
    mask = torch.tensor([True, False, True, False], dtype=torch.bool)

    filt = Filter(
        slices=[None, None],
        mask=mask,
        virtual_shape=base.shape,
    )

    ft = FilteredTensor.from_unmasked_value(value=base, filter_obj=filt)

    expected_inner = base[mask]
    assert ft.shape == base.shape
    assert ft.value.shape == expected_inner.shape
    assert torch.allclose(ft.value, expected_inner)
    assert torch.equal(ft.filter.mask, mask)
    assert ft.filter.virtual_shape == base.shape


# ---------------------------------------------------------------------------
# mask_by_other
# ---------------------------------------------------------------------------


def test_mask_by_other_with_filter_intersects_outer_masks():
    ft, base, mask1 = make_filtered_dense(
        n_docs=5,
        n_features=4,
        true_idx=(0, 2, 4),
    )
    mask2 = torch.tensor([True, True, False, False, True], dtype=torch.bool)

    other_filter = Filter(
        slices=[None, None],
        mask=mask2,
        virtual_shape=base.shape,
    )

    ft2 = ft.mask_by_other(other_filter, return_ft=True)

    # outer mask should be intersection
    expected_outer_mask = mask1 & mask2
    assert torch.equal(ft2.filter.mask, expected_outer_mask)

    # value should correspond to base[intersection]
    expected_value = base[expected_outer_mask]
    assert ft2.value.shape == expected_value.shape
    assert torch.allclose(ft2.value, expected_value)


def test_mask_by_other_with_tensor_presliced_value_like():
    ft, base, mask = make_filtered_dense(
        n_docs=6,
        n_features=2,
        true_idx=(1, 2, 4),
    )
    # ft.value shape: (#mask_true, 2)
    # Make a mask over "inner docs" (i.e., value_like=True semantics)
    inner_mask = torch.tensor([True, False, True], dtype=torch.bool)

    ft2 = ft.mask_by_other(
        other=inner_mask,
        return_ft=True,
        presliced=True,
        value_like=True,
    )

    # inner mask is applied directly to value
    expected_inner = ft.value[inner_mask]
    assert torch.allclose(ft2.value, expected_inner)

    # Outer mask is remapped via masked_scatter into the original mask positions
    # Build expected outer mask manually.
    outer_mask = torch.zeros_like(mask)
    outer_mask[mask] = inner_mask
    assert torch.equal(ft2.filter.mask, outer_mask)


# ---------------------------------------------------------------------------
# filter_inactive_docs (dense + sparse)
# ---------------------------------------------------------------------------


def test_filter_inactive_docs_dense():
    # start with outer mask == all docs active
    base = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # doc 0 inactive
            [1.0, 0.0, 0.0],  # doc 1 active
            [0.0, 0.0, 0.0],  # doc 2 inactive
            [0.0, 2.0, 0.0],  # doc 3 active
            [0.0, 0.0, 0.0],  # doc 4 inactive
        ]
    )
    mask = torch.ones(base.shape[0], dtype=torch.bool)

    ft = FilteredTensor.from_unmasked_value(value=base, filter_obj=mask)

    ft_active = ft.filter_inactive_docs()

    expected_outer_mask = torch.tensor(
        [False, True, False, True, False], dtype=torch.bool
    )
    expected_value = base[expected_outer_mask]

    assert torch.equal(ft_active.filter.mask, expected_outer_mask)
    assert torch.allclose(ft_active.value, expected_value)


def test_filter_inactive_docs_sparse():
    # Use the sparse helper; it starts with outer mask for docs 1, 2, 4,
    # then filter_inactive_docs should detect which of those are actually present.
    ft, outer_dense, mask = make_filtered_sparse_with_mask()

    # All docs currently in `value` are active by construction, but we can
    # zero out one inner doc to test the behavior.
    dense_inner = ft.value.to_dense()
    # Zero out the middle doc in inner coordinates
    dense_inner[1] = 0.0
    ft_zeroed = FilteredTensor.from_value_and_mask(
        value=dense_inner.to_sparse_coo(),
        mask_obj=ft.filter.mask,
    )

    ft_active = ft_zeroed.filter_inactive_docs()

    # We started with outer docs (1, 2, 4) active, but doc 2 had its row zeroed.
    expected_outer_mask = torch.tensor(
        [False, True, False, False, True],
        dtype=torch.bool,
    )
    reconstructed_dense = ft_active.to_sparse_unfiltered().to_dense()

    assert torch.equal(ft_active.filter.mask, expected_outer_mask)
    # The reconstructed dense tensor should match `outer_dense` with doc 2 zeroed out
    expected_outer_dense = outer_dense.clone() * mask.unsqueeze(-1)
    expected_outer_dense[2] = 0.0
    assert torch.allclose(reconstructed_dense, expected_outer_dense)


# ---------------------------------------------------------------------------
# Sparse/unfiltered conversions
# ---------------------------------------------------------------------------


def test_to_sparse_unfiltered_roundtrip_dense():
    """
    Build a sparse FilteredTensor and check that to_sparse_unfiltered()
    produces a tensor whose dense representation matches the expected outer
    dense tensor.
    """
    ft, outer_dense, mask = make_filtered_sparse_with_mask()

    sparse_unfiltered = ft.to_sparse_unfiltered()
    dense_roundtrip = sparse_unfiltered.to_dense()
    outer_dense = outer_dense * mask.unsqueeze(-1)
    assert dense_roundtrip.shape == outer_dense.shape
    assert torch.allclose(dense_roundtrip, outer_dense)


def test_indices_and_nonzero_agree_with_dense():
    ft, outer_dense, mask = make_filtered_sparse_with_mask()

    sparse_unfiltered = ft.to_sparse_unfiltered()

    # indices() should match the nonzero positions of the outer dense tensor
    dense_nz = (outer_dense * mask.unsqueeze(-1)).nonzero(as_tuple=False).T  # (2, nnz)
    ft_indices = ft.indices()

    assert ft_indices.shape == dense_nz.shape, (
        f"ft_indices.shape: {ft_indices.shape}, dense_nz.shape: {dense_nz.shape}, \n\n::"
        f"ft_indices: {ft_indices}, \n\n::"
        f"dense_nz: {dense_nz}"
    )
    # Order of non-zero entries from to_sparse/to_dense is stable here
    assert torch.equal(ft_indices, dense_nz), 2

    # nonzero() should give the same coordinates
    ft_nz = ft.to_dense().nonzero()
    assert torch.equal(ft_nz, dense_nz), 3


# ---------------------------------------------------------------------------
# Device / dtype quirks
# ---------------------------------------------------------------------------


def test_to_moves_both_value_and_filter():
    ft, *_ = make_filtered_dense(n_docs=4, n_features=2, true_idx=(1, 3))

    ft2 = ft.to(torch.device("cpu"), dtype=torch.float64)

    assert ft2.value.dtype == torch.float64
    assert ft2.filter.mask.device == torch.device("cpu")
    assert torch.allclose(ft2.value.to(torch.float32), ft.value)
    assert torch.equal(ft2.filter.mask, ft.filter.mask)


def test_shape_and_is_sparse_properties():
    ft_dense, base_dense, _ = make_filtered_dense()
    assert not ft_dense.is_sparse
    assert ft_dense.shape == base_dense.shape

    ft_sparse, outer_dense = make_filtered_sparse()
    assert ft_sparse.is_sparse
    assert ft_sparse.shape == outer_dense.shape


# ---------------------------------------------------------------------------
# Known-issue marker for to_sparse on dense values
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="FilteredTensor.to_sparse() assumes the underlying value is already sparse."
)
def test_to_sparse_from_dense_not_yet_implemented():
    """
    This currently fails because FilteredTensor.to_sparse() calls .indices()
    and .values() on a dense tensor.

    Once to_sparse() is fixed to handle dense inputs, this test can be turned
    into a normal passing test.
    """
    base = torch.arange(12.0).reshape(4, 3)
    mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    ft = FilteredTensor.from_unmasked_value(value=base, filter_obj=mask)

    # This currently raises AttributeError (no .indices on dense tensor)
    _ = ft.to_sparse()


if __name__ == "__main__":
    test_indices_and_nonzero_agree_with_dense()
    test_filter_inactive_docs_sparse()
    test_to_sparse_unfiltered_roundtrip_dense()
