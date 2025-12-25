import os
from enum import Enum
from pathlib import Path

import torch
import zstd
from safetensors import safe_open
from safetensors.torch import load, save


class CompressionType(str, Enum):
    NONE = "none"
    ZSTD = "zstd"

    def compress(self, obj):
        if self == CompressionType.NONE:
            return obj
        elif self == CompressionType.ZSTD:
            return zstd.compress(obj)
        raise ValueError(f"Unknown compression type: {self}")

    def decompress(self, obj):
        if self == CompressionType.NONE:
            return obj
        elif self == CompressionType.ZSTD:
            return zstd.decompress(obj)
        raise ValueError(f"Unknown compression type: {self}")


def save_file_compressed(
    tensors: dict[str, torch.Tensor],
    filename: str | os.PathLike,
    compression: CompressionType,
    metadata: dict[str, str] | None = None,
):
    file = Path(filename)
    if file.exists():
        raise ValueError(f"File {file} already exists!")
    b = save(tensors, metadata=metadata)
    c = compression.compress(b)
    file.write_bytes(c)


def load_file_compressed(
    filename: str | os.PathLike,
    compression: CompressionType,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    file = Path(filename)
    b = compression.decompress(file.read_bytes())
    return {k: v.to(device) for k, v in load(b).items()}


def load_file_with_metadata(
    filename: str | os.PathLike,
    device: torch.device | str = "cpu",
):
    result = {}

    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.offset_keys():
            result[k] = f.get_tensor(k)
        metadata = f.metadata()
    return result, metadata


def main():
    td = Path.cwd() / "testdata" / "compressed_safetensors"
    td.mkdir(exist_ok=True, parents=True)
    tdp = td / "test.safetensors"
    if tdp.exists():
        tdp.unlink()
    t1 = torch.randn(44, 42)
    t2 = torch.randn(4, 5)
    save_file_compressed(
        {"t1": t1, "t2": t2}, tdp, metadata={"t1": "t2221", "t2": "t2"}
    )

    loaded = load_file_compressed(tdp, device="cpu")
    assert torch.allclose(t1, loaded["t1"])
    assert torch.allclose(t2, loaded["t2"])
    print("ok")
    print(loaded)


if __name__ == "__main__":
    main()
