from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from saeco.data.dict_batch import DictBatch

_DEFAULT_TOKEN_KEYS: tuple[str, ...] = ("tokens", "input_ids", "ids")


def extract_token_tensor(
    x: Tensor | DictBatch,
    *,
    keys: Iterable[str] = _DEFAULT_TOKEN_KEYS,
) -> Tensor:
    """
    Extract a token-id Tensor from either:
      - a Tensor directly, or
      - a DictBatch containing a token-id field.
    """
    if isinstance(x, Tensor):
        return x

    for key in keys:
        try:
            t = x[key]
            if isinstance(t, Tensor):
                return t
        except Exception:
            continue

    raise KeyError(
        f"Could not find token id tensor in DictBatch. Tried keys: {tuple(keys)}"
    )
