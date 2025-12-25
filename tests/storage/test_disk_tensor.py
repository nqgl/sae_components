from pathlib import Path
import pytest
import torch

from saeco.data.storage.disk_tensor import (
    DiskTensor,
    DiskTensorMetadata,
    _numel_from_shape,
)
from saeco.data.storage.compressed_safetensors import CompressionType


@pytest.fixture
def tmp_tensor_path(tmp_path: Path) -> Path:
    return tmp_path / "test_tensor"


class TestDiskTensorMetadata:
    def test_dtype_property(self):
        meta = DiskTensorMetadata(shape=[10, 10], dtype_str="torch.float32")
        assert meta.dtype == torch.float32

    def test_dtype_setter(self):
        meta = DiskTensorMetadata(shape=[10, 10], dtype_str="torch.float32")
        meta.dtype = torch.int64
        assert meta.dtype_str == "torch.int64"
        assert meta.dtype == torch.int64

    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("torch.float32", torch.float32),
            ("torch.int64", torch.int64),
            ("torch.bool", torch.bool),
            ("torch.float16", torch.float16),
            ("torch.bfloat16", torch.bfloat16),
        ],
    )
    def test_dtype_roundtrip(self, dtype_str: str, expected: torch.dtype):
        meta = DiskTensorMetadata(shape=[5], dtype_str=dtype_str)
        assert meta.dtype == expected


class TestNumelFromShape:
    @pytest.mark.parametrize(
        "shape,expected",
        [
            ([10], 10),
            ([10, 10], 100),
            ([2, 3, 4], 24),
            ([1], 1),
            ([0], 0),
            ([10, 0, 5], 0),
        ],
    )
    def test_various_shapes(self, shape: list[int], expected: int):
        assert _numel_from_shape(shape) == expected

    def test_empty_shape(self):
        assert _numel_from_shape([]) == 0


class TestDiskTensorCreate:
    def test_create_basic(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10, 10), torch.float32)

        assert dt.path == tmp_tensor_path
        assert dt.metadata.shape == [10, 10]
        assert dt.metadata.dtype == torch.float32
        assert dt.finalized is False

    def test_create_with_compression(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(
            tmp_tensor_path, (5, 5), torch.int64, compression=CompressionType.ZSTD
        )
        assert dt.metadata.compression == CompressionType.ZSTD

    def test_create_raises_if_path_exists(self, tmp_tensor_path: Path):
        tmp_tensor_path.touch()

        with pytest.raises(ValueError, match="already exists"):
            DiskTensor.create(tmp_tensor_path, (10,), torch.float32)

    def test_create_raises_if_safetensors_exists(self, tmp_tensor_path: Path):
        tmp_tensor_path.with_suffix(".safetensors").touch()

        with pytest.raises(ValueError, match="already exists"):
            DiskTensor.create(tmp_tensor_path, (10,), torch.float32)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.int64,
            torch.bool,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_create_various_dtypes(self, tmp_path: Path, dtype: torch.dtype):
        path = tmp_path / f"tensor_{dtype}"
        dt = DiskTensor.create(path, (10,), dtype)

        assert dt.metadata.dtype == dtype
        assert dt.tensor.dtype == dtype


class TestDiskTensorTensorAccess:
    def test_tensor_creates_file(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10, 10), torch.float32)

        assert not tmp_tensor_path.exists()
        _ = dt.tensor
        assert tmp_tensor_path.exists()

    def test_tensor_write_read(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10, 10), torch.int64)
        expected = torch.arange(100).reshape(10, 10)
        dt.tensor[:] = expected

        assert torch.equal(dt.tensor, expected)

    def test_tensor_persistence(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (5, 5), torch.float32)
        expected = torch.randn(5, 5)
        dt.tensor[:] = expected

        # Clear cached property
        del dt.tensor

        assert torch.allclose(dt.tensor, expected)

    def test_tensor_shape(self, tmp_tensor_path: Path):
        shape = (3, 4, 5)
        dt = DiskTensor.create(tmp_tensor_path, shape, torch.float32)

        assert dt.tensor.shape == shape


class TestDiskTensorFinalize:
    def test_finalize_creates_safetensors(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10,), torch.float32)
        dt.tensor[:] = torch.arange(10, dtype=torch.float32)
        dt.finalize()

        assert tmp_tensor_path.with_suffix(".safetensors").exists()
        assert tmp_tensor_path.with_suffix(".metadata").exists()
        assert not tmp_tensor_path.exists()  # raw file deleted

    def test_finalize_sets_flag(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10,), torch.float32)
        _ = dt.tensor

        assert not dt.finalized
        dt.finalize()
        assert dt.finalized

    def test_finalize_preserves_data(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10, 10), torch.int64)
        expected = torch.arange(100).reshape(10, 10)
        dt.tensor[:] = expected
        dt.finalize()

        # Reopen and verify
        dt2 = DiskTensor.open(tmp_tensor_path)
        assert torch.equal(dt2.tensor, expected)


class TestDiskTensorOpen:
    def test_open_finalized(self, tmp_tensor_path: Path):
        # Create and finalize
        dt = DiskTensor.create(tmp_tensor_path, (5, 5), torch.float32)
        expected = torch.randn(5, 5)
        dt.tensor[:] = expected
        dt.finalize()

        # Open
        dt2 = DiskTensor.open(tmp_tensor_path)

        assert dt2.finalized
        assert torch.allclose(dt2.tensor, expected)

    def test_open_reads_metadata(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (3, 4, 5), torch.int64)
        dt.tensor[:] = 0
        dt.finalize()

        dt2 = DiskTensor.open(tmp_tensor_path)

        assert dt2.metadata.shape == [3, 4, 5]
        assert dt2.metadata.dtype == torch.int64


class TestDiskTensorProperties:
    def test_storage_shape(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (2, 3, 4), torch.float32)
        assert dt.storage_shape == [2, 3, 4]

    def test_numel(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (2, 3, 4), torch.float32)
        assert dt.numel == 24

    def test_nbytes(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10, 10), torch.float32)
        assert dt.nbytes == 10 * 10 * 4  # float32 = 4 bytes

    def test_nbytes_int64(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10, 10), torch.int64)
        assert dt.nbytes == 10 * 10 * 8  # int64 = 8 bytes

    def test_metadata_path(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10,), torch.float32)
        assert dt.metadata_path == tmp_tensor_path.with_suffix(".metadata")


class TestDiskTensorOpenDiskTensor:
    def test_open_nonexistent_without_create_raises(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10,), torch.float32)

        with pytest.raises(FileNotFoundError):
            dt.open_disk_tensor(create=False)

    def test_open_existing_with_create_raises(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10,), torch.float32)
        _ = dt.tensor  # creates file

        # Clear cache to force re-open
        del dt.tensor

        with pytest.raises(FileExistsError):
            dt.open_disk_tensor(create=True)


class TestDiskTensorRoundtrip:
    """End-to-end tests for create -> write -> finalize -> open cycle."""

    @pytest.mark.parametrize(
        "shape",
        [
            (10,),
            (5, 5),
            (2, 3, 4),
            (1, 1, 1, 1),
        ],
    )
    def test_various_shapes(self, tmp_path: Path, shape: tuple[int, ...]):
        path = tmp_path / "tensor"
        dt = DiskTensor.create(path, shape, torch.float32)
        expected = torch.randn(shape)
        dt.tensor[:] = expected
        dt.finalize()

        dt2 = DiskTensor.open(path)
        assert torch.allclose(dt2.tensor, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_various_dtypes(self, tmp_path: Path, dtype: torch.dtype):
        path = tmp_path / "tensor"
        dt = DiskTensor.create(path, (10, 10), dtype)

        if dtype in (torch.float32, torch.float16, torch.bfloat16):
            expected = torch.randn(10, 10, dtype=dtype)
        else:
            expected = torch.randint(0, 100, (10, 10), dtype=dtype)

        dt.tensor[:] = expected
        dt.finalize()

        dt2 = DiskTensor.open(path)
        assert torch.equal(dt2.tensor, expected)

    def test_bool_dtype(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (10, 10), torch.bool)
        expected = torch.randint(0, 2, (10, 10), dtype=torch.bool)
        dt.tensor[:] = expected
        dt.finalize()

        dt2 = DiskTensor.open(tmp_tensor_path)
        assert torch.equal(dt2.tensor, expected)


class TestDiskTensorEdgeCases:
    def test_scalar_like_shape(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (1,), torch.float32)
        dt.tensor[:] = torch.tensor([42.0])
        dt.finalize()

        dt2 = DiskTensor.open(tmp_tensor_path)
        assert dt2.tensor.item() == 42.0

    def test_large_tensor(self, tmp_tensor_path: Path):
        # 1M elements - not huge but verifies chunked I/O works
        shape = (1000, 1000)
        dt = DiskTensor.create(tmp_tensor_path, shape, torch.float32)
        expected = torch.randn(shape)
        dt.tensor[:] = expected
        dt.finalize()

        dt2 = DiskTensor.open(tmp_tensor_path)
        assert torch.allclose(dt2.tensor, expected)

    def test_tensor_shared_memory(self, tmp_tensor_path: Path):
        """Verify that shared=True allows cross-ref erence writes."""
        dt = DiskTensor.create(tmp_tensor_path, (10,), torch.int64)
        tensor1 = dt.tensor

        # Get another view via reopening
        del dt.tensor
        tensor2 = dt.tensor

        tensor1[0] = 999
        assert tensor2[0] == 999

    def test_empty_tensor(self, tmp_tensor_path: Path):
        dt = DiskTensor.create(tmp_tensor_path, (0,), torch.float32)
        dt.tensor  # trigger creation
        dt.finalize()

        dt2 = DiskTensor.open(tmp_tensor_path)
        assert dt2.tensor.numel() == 0
