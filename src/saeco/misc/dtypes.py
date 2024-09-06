import torch

S2D = {
    "float32": torch.float32,
    "int64": torch.int64,
    "bool": torch.bool,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
D2S = {v: k for k, v in S2D.items()}


def str_to_dtype(dtype: str) -> torch.dtype:
    if "torch." in dtype:
        return S2D[dtype.replace("torch.", "")]
    return S2D[dtype]
