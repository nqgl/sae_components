from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Final

import torch
from pydantic import BaseModel
from torch import Tensor

_MAX_TENSOR_BYTES: Final[int] = 8 * 1024 * 1024
_MAX_TENSOR_SAMPLES: Final[int] = 4096


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _slugify(s: str, *, max_len: int = 80) -> str:
    # Keep alnum plus a handful of separators; everything else becomes "_".
    out: list[str] = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_.-")
    return (slug[:max_len] or "call").lower()


def _tensor_hash(t: Tensor) -> str:
    """
    Deterministic content hash for use in cache keys.

    - Always hashes via CPU for stable bytes.
    - Sparse tensors hash (indices, values).
    - Dense tensors hash full bytes up to _MAX_TENSOR_BYTES, else hash a fixed sample.
    """
    t = t.detach()

    if t.is_sparse:
        coo = t.coalesce().cpu()
        idx = coo.indices().contiguous()
        val = coo.values().contiguous()
        h = hashlib.sha256()
        h.update(str(idx.dtype).encode("utf-8"))
        h.update(str(tuple(idx.shape)).encode("utf-8"))
        h.update(idx.numpy().tobytes())
        h.update(str(val.dtype).encode("utf-8"))
        h.update(str(tuple(val.shape)).encode("utf-8"))
        h.update(val.numpy().tobytes())
        return h.hexdigest()

    cpu = t.cpu().contiguous()
    nbytes = cpu.numel() * cpu.element_size()
    if nbytes <= _MAX_TENSOR_BYTES:
        return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()

    flat = cpu.flatten()
    n = flat.numel()
    if n == 0:
        return "0" * 64

    k = min(_MAX_TENSOR_SAMPLES, n)
    sample_idx = torch.linspace(0, n - 1, steps=k).round().to(dtype=torch.long)
    sample = flat.index_select(0, sample_idx).contiguous()
    return hashlib.sha256(sample.numpy().tobytes()).hexdigest()


def _canonicalize(obj: Any) -> Any:
    """
    Convert an arbitrary Python object into a JSON-serializable, deterministic structure.
    Used only for hashing/cache keys.
    """
    match obj:
        case None:
            return None
        case bool():
            return obj
        case int():
            return obj
        case float():
            # JSON doesn't represent NaN/Inf portably.
            if obj != obj:
                return {"__type__": "float", "value": "NaN"}
            if obj == float("inf"):
                return {"__type__": "float", "value": "Infinity"}
            if obj == float("-inf"):
                return {"__type__": "float", "value": "-Infinity"}
            return obj
        case str():
            return obj
        case bytes():
            return {"__type__": "bytes", "b64": base64.b64encode(obj).decode("ascii")}
        case Path():
            return {"__type__": "path", "value": str(obj)}
        case Tensor():
            return {
                "__type__": "tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "sparse": bool(obj.is_sparse),
                "hash": _tensor_hash(obj),
            }
        case tuple() | list():
            return [_canonicalize(x) for x in obj]
        case set() | frozenset():
            # Order-independent, deterministic.
            items = [_canonicalize(x) for x in obj]
            return sorted(items, key=_json_dumps)
        case dict():
            # Sort keys deterministically by string form.
            return {
                str(k): _canonicalize(v)
                for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
            }
        case BaseModel():
            return {
                "__type__": f"pydantic:{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "value": obj.model_dump(mode="json"),
            }
        case _:
            return {
                "__type__": f"repr:{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "repr": repr(obj),
            }


def func_identity(func: Any, *, name_override: str | None = None) -> str:
    name = name_override or getattr(func, "__qualname__", getattr(func, "__name__", "call"))
    module = getattr(func, "__module__", None)
    return f"{module}.{name}" if module else name


def call_cache_digest(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name_override: str | None = None,
    version: Any | None = None,
) -> str:
    payload: dict[str, Any] = {
        "func": func_identity(func, name_override=name_override),
        "args": _canonicalize(args),
        "kwargs": _canonicalize(kwargs),
    }
    if version is not None:
        payload["version"] = _canonicalize(version)
    return _sha256_str(_json_dumps(payload))


def call_cache_key(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name_override: str | None = None,
    version: Any | None = None,
    prefix: str | None = None,
) -> str:
    """
    Filesystem-safe cache key.

    Includes:
      - a human-ish slug prefix (short)
      - a full sha256 digest suffix

    The slug is for browsing/debugging; the digest is the real identity.
    """
    digest = call_cache_digest(
        func, args, kwargs, name_override=name_override, version=version
    )
    slug = _slugify(prefix or func_identity(func, name_override=name_override))
    return f"{slug}-{digest}"
