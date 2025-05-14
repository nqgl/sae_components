import torch

DTYPES_DEDUPLICATED = {
    torch.float32,
    torch.float,
    torch.float64,
    torch.double,
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.half,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.short,
    torch.int32,
    torch.int,
    torch.int64,
    torch.long,
    torch.complex32,
    torch.complex64,
    torch.chalf,
    torch.cfloat,
    torch.complex128,
    torch.cdouble,
    torch.quint8,
    torch.qint8,
    torch.qint32,
    torch.bool,
    torch.quint4x2,
    torch.quint2x4,
    torch.bits1x8,
    torch.bits2x4,
    torch.bits4x2,
    torch.bits8,
    torch.bits16,
}
S2D = {str(d): d for d in DTYPES_DEDUPLICATED}
D2S = {d: str(d) for d in DTYPES_DEDUPLICATED}


def str_to_dtype(dtype: str, strict: bool = False) -> torch.dtype:
    if dtype not in S2D:
        if strict:
            raise ValueError(f"Unsupported dtype string: {dtype}")
        else:
            return str_to_dtype(f"torch.{dtype}", strict=True)
    return S2D[dtype]


def main():
    for d in DTYPES_DEDUPLICATED:
        assert str_to_dtype(str(d)) == d


if __name__ == "__main__":
    main()
