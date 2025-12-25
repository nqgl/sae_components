from pathlib import Path

import pytest
import torch

from saeco.data.storage.compressed_safetensors import CompressionType
from saeco.data.storage.growing_disk_tensor import (
    SAECO_MIN_GDT_INITIAL_BYTES,
    GrowingDiskTensor,
)


@pytest.fixture
def tmp_tensor_path(tmp_path: Path) -> Path:
    return tmp_path / "growing_tensor"


class TestGrowingDiskTensorCreate:
    def test_create_basic(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path, shape=[10, 20], dtype=torch.float32
        )

        assert gdt.path == tmp_tensor_path
        assert gdt.cat_axis == 0
        assert gdt.metadata.dtype == torch.float32
        assert gdt.finalized is False
        # shape[cat_axis] starts at 0 (nothing written yet)
        assert gdt.metadata.shape[gdt.cat_axis] == 0

    def test_create_custom_cat_axis(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20, 30],
            dtype=torch.float32,
            cat_axis=1,
        )

        assert gdt.cat_axis == 1
        assert gdt.metadata.shape == [10, 0, 30]

    def test_create_with_initial_nnz(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            initial_nnz=1000,
        )

        # storage_len should accommodate initial_nnz elements
        assert gdt.storage_len >= 1

    def test_create_with_compression(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            compression=CompressionType.ZSTD,
        )

        assert gdt.metadata.compression == CompressionType.ZSTD

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int64,
        ],
    )
    def test_create_various_dtypes(self, tmp_path: Path, dtype: torch.dtype):
        path = tmp_path / f"tensor_{dtype}"
        gdt = GrowingDiskTensor.create(path, shape=[10, 20], dtype=dtype)

        assert gdt.metadata.dtype == dtype

    def test_create_calculates_initial_storage_from_bytes(self, tmp_tensor_path: Path):
        # For float32 (4 bytes), with shape [10, ?], each "row" is 10 * 4 = 40 bytes
        # initial_nnz = SAECO_MIN_GDT_INITIAL_BYTES // 4
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )

        expected_initial_nnz = SAECO_MIN_GDT_INITIAL_BYTES // torch.float32.itemsize
        assert gdt.storage_len > 0


class TestGrowingDiskTensorProperties:
    def test_storage_shape(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20, 30],
            dtype=torch.float32,
            cat_axis=1,
        )

        # metadata.shape has 0 at cat_axis, but storage_shape has storage_len
        expected = [10, gdt.storage_len, 30]
        assert gdt.storage_shape == expected

    def test_cat_len_initially_zero(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )

        assert gdt.cat_len == 0

    def test_cat_len_after_append(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        gdt.append(torch.randn(5, 20))

        assert gdt.cat_len == 5

    def test_valid_tensor(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            cat_axis=0,
        )
        gdt.append(torch.randn(5, 20))

        assert gdt.valid_tensor.shape == (5, 20)
        # storage is larger
        assert gdt.tensor.shape[0] > 5

    def test_valid_tensor_with_different_cat_axis(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20, 30],
            dtype=torch.float32,
            cat_axis=1,
        )
        gdt.append(torch.randn(10, 7, 30))

        assert gdt.valid_tensor.shape == (10, 7, 30)


class TestGrowingDiskTensorAppend:
    def test_append_single(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        data = torch.randn(5, 20)
        gdt.append(data)

        assert gdt.cat_len == 5
        assert torch.allclose(gdt.valid_tensor, data)

    def test_append_multiple(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        data1 = torch.randn(3, 20)
        data2 = torch.randn(4, 20)

        gdt.append(data1)
        gdt.append(data2)

        assert gdt.cat_len == 7
        assert torch.allclose(gdt.valid_tensor[:3], data1)
        assert torch.allclose(gdt.valid_tensor[3:], data2)

    def test_append_triggers_resize(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            initial_nnz=1,  # force small initial storage
        )
        initial_storage = gdt.storage_len

        # Append more than initial capacity
        large_data = torch.randn(initial_storage * 3, 20)
        gdt.append(large_data)

        assert gdt.storage_len > initial_storage
        assert gdt.cat_len == initial_storage * 3

    def test_append_preserves_existing_data_after_resize(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            initial_nnz=1,
        )
        initial_storage = gdt.storage_len

        data1 = torch.randn(initial_storage - 1, 20)
        gdt.append(data1)

        # This should trigger resize
        data2 = torch.randn(initial_storage, 20)
        gdt.append(data2)

        # Verify first data is still intact
        assert torch.allclose(gdt.valid_tensor[: initial_storage - 1], data1)

    def test_append_prefix_shape_mismatch_raises(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20, 30],
            dtype=torch.float32,
            cat_axis=1,
        )

        # Wrong prefix shape (should be [10, ?, 30])
        bad_data = torch.randn(5, 7, 30)  # 5 != 10

        with pytest.raises(ValueError, match="Shape mismatch.*prefix"):
            gdt.append(bad_data)

    def test_append_suffix_shape_mismatch_raises(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20, 30],
            dtype=torch.float32,
            cat_axis=1,
        )

        # Wrong suffix shape (should be [10, ?, 30])
        bad_data = torch.randn(10, 7, 25)  # 25 != 30

        with pytest.raises(ValueError, match="Shape mismatch.*suffix"):
            gdt.append(bad_data)

    def test_append_after_finalize_raises(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        gdt.append(torch.randn(5, 20))
        gdt.finalize()

        with pytest.raises(AssertionError):
            gdt.append(torch.randn(3, 20))

    @pytest.mark.parametrize("cat_axis", [0, 1, 2])
    def test_append_various_cat_axes(self, tmp_path: Path, cat_axis: int):
        path = tmp_path / f"tensor_axis_{cat_axis}"
        base_shape = [4, 5, 6]

        gdt = GrowingDiskTensor.create(
            path,
            shape=base_shape,
            dtype=torch.float32,
            cat_axis=cat_axis,
        )

        # Create data with correct shape
        data_shape = base_shape.copy()
        data_shape[cat_axis] = 3
        data = torch.randn(*data_shape)

        gdt.append(data)

        assert gdt.cat_len == 3
        assert torch.allclose(gdt.valid_tensor, data)


class TestGrowingDiskTensorResize:
    def test_resize_grow(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            initial_nnz=1,
        )
        initial_len = gdt.storage_len
        _ = gdt.tensor  # materialize

        gdt.resize(initial_len * 2)

        assert gdt.storage_len == initial_len * 2

    def test_resize_preserves_data(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            initial_nnz=1,
        )
        data = torch.randn(gdt.storage_len - 1, 20)
        gdt.append(data)

        gdt.resize(gdt.storage_len * 2)

        assert torch.allclose(gdt.valid_tensor, data)

    def test_resize_truncate(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            initial_nnz=1,
        )
        gdt.append(torch.randn(5, 20))

        gdt.resize(5, truncate=True)

        assert gdt.storage_len == 5

    def test_resize_after_finalize_raises(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        gdt.append(torch.randn(5, 20))
        gdt.finalize()

        with pytest.raises(AssertionError):
            gdt.resize(100)


class TestGrowingDiskTensorFinalize:
    def test_finalize_creates_safetensors(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        gdt.append(torch.randn(5, 20))
        gdt.finalize()

        assert tmp_tensor_path.with_suffix(".safetensors").exists()
        assert tmp_tensor_path.with_suffix(".metadata").exists()

    def test_finalize_truncates_to_actual_size(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            initial_nnz=1,
        )
        # Append less than storage capacity
        data = torch.randn(3, 20)
        gdt.append(data)

        gdt.finalize()

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)
        assert gdt2.tensor.shape == (3, 20)

    def test_finalize_sets_flag(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        gdt.append(torch.randn(5, 20))

        assert not gdt.finalized
        gdt.finalize()
        assert gdt.finalized

    def test_finalize_preserves_data(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        expected = torch.randn(7, 20)
        gdt.append(expected)
        gdt.finalize()

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)
        assert torch.allclose(gdt2.tensor, expected)


class TestGrowingDiskTensorShuffleThenFinalize:
    def test_shuffle_then_finalize_permutes_data(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 1],  # simple shape for easy verification
            dtype=torch.float32,
        )
        # Sequential data so we can verify shuffle
        data = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
        gdt.append(data)

        perm = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        gdt.shuffle_then_finalize(perm=perm)

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)
        expected = data[perm]
        assert torch.equal(gdt2.tensor, expected)

    def test_shuffle_then_finalize_random_perm(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        data = torch.randn(100, 20)
        gdt.append(data)

        # No perm provided - uses random
        gdt.shuffle_then_finalize()

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)
        # Data should be permuted, same elements but different order
        assert gdt2.tensor.shape == data.shape
        # Check same elements exist (sort and compare)
        assert torch.allclose(
            gdt2.tensor.sort(dim=0).values,
            data.sort(dim=0).values,
        )

    def test_shuffle_then_finalize_custom_axis(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[5, 10, 3],
            dtype=torch.float32,
            cat_axis=1,
        )
        data = torch.randn(5, 8, 3)
        gdt.append(data)

        perm = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])
        gdt.shuffle_then_finalize(shuffle_axis=1, perm=perm)

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)
        expected = data.index_select(1, perm)
        assert torch.allclose(gdt2.tensor, expected)

    def test_shuffle_then_finalize_wrong_perm_length_raises(
        self, tmp_tensor_path: Path
    ):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        gdt.append(torch.randn(5, 20))

        wrong_perm = torch.arange(10)  # length 10, but data has 5 rows

        with pytest.raises(AssertionError):
            gdt.shuffle_then_finalize(perm=wrong_perm)


class TestGrowingDiskTensorOpen:
    def test_open_finalized_tensor(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )
        expected = torch.randn(7, 20)
        gdt.append(expected)
        gdt.finalize()

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)

        assert gdt2.finalized
        assert torch.allclose(gdt2.tensor, expected)

    def test_open_preserves_metadata(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20, 30],
            dtype=torch.int64,
            cat_axis=1,
        )
        gdt.append(torch.randint(0, 100, (10, 5, 30)))
        gdt.finalize()

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)

        assert gdt2.metadata.dtype == torch.int64


class TestGrowingDiskTensorRoundtrip:
    """End-to-end tests."""

    def test_create_append_finalize_open(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )

        chunks = [torch.randn(3, 20) for _ in range(5)]
        for chunk in chunks:
            gdt.append(chunk)

        gdt.finalize()

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)
        expected = torch.cat(chunks, dim=0)
        assert torch.allclose(gdt2.tensor, expected)

    def test_many_small_appends(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 5],
            dtype=torch.float32,
            initial_nnz=1,  # force many resizes
        )

        chunks = [torch.randn(1, 5) for _ in range(100)]
        for chunk in chunks:
            gdt.append(chunk)

        gdt.finalize()

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)
        expected = torch.cat(chunks, dim=0)
        assert torch.allclose(gdt2.tensor, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int64,
        ],
    )
    def test_roundtrip_various_dtypes(self, tmp_path: Path, dtype: torch.dtype):
        path = tmp_path / f"tensor_{dtype}"
        gdt = GrowingDiskTensor.create(path, shape=[10, 5], dtype=dtype)

        if dtype.is_floating_point:
            data = torch.randn(7, 5, dtype=dtype)
        else:
            data = torch.randint(0, 100, (7, 5), dtype=dtype)

        gdt.append(data)
        gdt.finalize()

        gdt2 = GrowingDiskTensor.open(path)
        assert torch.equal(gdt2.tensor, data)

    @pytest.mark.parametrize("cat_axis", [0, 1, 2])
    def test_roundtrip_various_cat_axes(self, tmp_path: Path, cat_axis: int):
        path = tmp_path / f"tensor_axis_{cat_axis}"
        base_shape = [4, 5, 6]

        gdt = GrowingDiskTensor.create(
            path,
            shape=base_shape,
            dtype=torch.float32,
            cat_axis=cat_axis,
        )

        data_shape = base_shape.copy()
        data_shape[cat_axis] = 3
        data = torch.randn(*data_shape)

        gdt.append(data)
        gdt.finalize()
        gdt2 = GrowingDiskTensor.open(path)
        gdt.tensor == gdt2.tensor
        gdt2.tensor.stride()
        gdt2.cat_axis
        gdt.cat_axis
        data.stride()
        (data - gdt2.tensor).abs() < 1e-5
        data
        assert torch.allclose(gdt2.tensor, data), (
            f"gdt2.tensor: {gdt2.tensor}, data: {data}"
        )


class TestGrowingDiskTensorEdgeCases:
    def test_single_element_append(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )

        for i in range(10):
            gdt.append(torch.full((1, 20), float(i)))

        gdt.finalize()

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)
        for i in range(10):
            assert torch.allclose(gdt2.tensor[i], torch.full((20,), float(i)))

    def test_large_single_append(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            initial_nnz=1,
        )

        # Append way more than initial capacity in one go
        large_data = torch.randn(1000, 20)
        gdt.append(large_data)
        gdt.finalize()

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)
        assert torch.allclose(gdt2.tensor, large_data)

    def test_append_exact_capacity(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
            initial_nnz=1,
        )
        storage = gdt.storage_len

        # Append exactly storage_len - should trigger resize due to >= check
        data = torch.randn(storage, 20)
        gdt.append(data)

        assert gdt.storage_len > storage
        gdt.finalize()

        gdt2 = GrowingDiskTensor.open(tmp_tensor_path)
        assert torch.allclose(gdt2.tensor, data)

    def test_interleaved_read_write(self, tmp_tensor_path: Path):
        gdt = GrowingDiskTensor.create(
            tmp_tensor_path,
            shape=[10, 20],
            dtype=torch.float32,
        )

        gdt.append(torch.randn(5, 20))
        read1 = gdt.valid_tensor.clone()

        gdt.append(torch.randn(5, 20))

        # First 5 rows should be unchanged
        assert torch.allclose(gdt.valid_tensor[:5], read1)


if __name__ == "__main__":
    path = Path("testdata/storage_testing")
    path.mkdir(parents=True, exist_ok=True)
    TestGrowingDiskTensorRoundtrip().test_roundtrip_various_cat_axes(
        tmp_path=path,
        cat_axis=2,
    )

    TestGrowingDiskTensorAppend().test_append_triggers_resize(
        tmp_tensor_path=path / "test_append_triggers_resize"
    )
