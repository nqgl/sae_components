from __future__ import annotations

import functools
import inspect
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_origin

from pydantic import BaseModel
from torch import Tensor

from saeco.data.dict_batch import DictBatch

from .cache_keys import call_cache_key
from .filtered import FilteredTensor

if TYPE_CHECKING:
    from saeco_research.evaluation.evaluation import Evaluation


def _is_subclass_safe(t: Any, base: type) -> bool:
    try:
        return isinstance(t, type) and issubclass(t, base)
    except Exception:
        return False


def _iter_union_members(ann: Any) -> tuple[Any, ...]:
    if ann is None or ann is inspect.Signature.empty:
        return ()
    if isinstance(ann, str):
        return ()
    if isinstance(ann, types.UnionType):
        return ann.__args__
    origin = get_origin(ann)
    if origin is None:
        return (ann,)
    if origin is types.UnionType:
        return get_args(ann)
    if origin is getattr(types, "UnionType", None):
        return get_args(ann)
    if origin is getattr(__import__("typing"), "Union"):
        return get_args(ann)
    return (ann,)


def _cache_dir_for_artifacts(artifacts: Any) -> Path:
    # DiskTensorCollection-like objects typically expose storage_dir.
    p = getattr(artifacts, "storage_dir", None)
    if isinstance(p, Path):
        return p
    p = getattr(artifacts, "path", None)
    if isinstance(p, Path):
        return p
    raise AttributeError("Artifacts collection does not expose a Path-like directory")


class CachedCalls:
    """
    Wrap an Evaluation and transparently cache call results into:
      - BMStorShelf for pydantic models
      - safetensors on disk for DictBatch
      - Artifacts collection for torch.Tensor

    The wrapper targets methods only; non-callables raise.
    """

    def __init__(self, raw: Evaluation):
        self.raw = raw
        if hasattr(self.raw, "raw"):
            raise AssertionError("Double-wrapping CachedCalls is not supported")

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.raw, name)
        if not callable(attr):
            raise AttributeError(f"{name} is not callable on {type(self.raw).__name__}")
        return self._wrap(attr, name_override=name)

    def _wrap(self, func: Any, name_override: str | None = None) -> Any:
        name_override = name_override or getattr(func, "__name__", "call")
        ann = inspect.signature(func).return_annotation
        ann_members = _iter_union_members(ann)
        version = getattr(func, "_version", None)

        wants_pydantic = any(_is_subclass_safe(t, BaseModel) for t in ann_members)
        wants_dictbatch = any(_is_subclass_safe(t, DictBatch) for t in ann_members)

        artifacts_dir = _cache_dir_for_artifacts(self.raw.artifact_store)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # --- Pydantic models -> BMStorShelf (versioned) ---
            if wants_pydantic:
                if self.raw.bmstore.has(func, args, kwargs):
                    return self.raw.bmstore.get(func, args, kwargs)
                value = func(*args, **kwargs)
                if not isinstance(value, BaseModel):
                    return value
                self.raw.bmstore.set(func, args, kwargs, value)
                return value

            # --- DictBatch -> safetensors dir ---
            if wants_dictbatch:
                key = call_cache_key(
                    func,
                    args=args,
                    kwargs=kwargs,
                    name_override=name_override,
                    version=version,
                    prefix=name_override,
                )
                path = artifacts_dir / "dict_batches" / key
                path.parent.mkdir(parents=True, exist_ok=True)

                if path.exists():
                    # Return type should be a DictBatch subclass
                    # with load_from_safetensors.
                    return_ann = ann if ann is not inspect.Signature.empty else None
                    if isinstance(return_ann, type) and issubclass(
                        return_ann, DictBatch
                    ):
                        return return_ann.load_from_safetensors(path)

                value = func(*args, **kwargs)
                if not isinstance(value, DictBatch):
                    return value
                value.save_as_safetensors(path)
                return value

            # --- Everything else: Tensor caching into artifacts collection ---
            key = call_cache_key(
                func,
                args=args,
                kwargs=kwargs,
                name_override=name_override,
                version=version,
                prefix=name_override,
            )

            if key in self.raw.artifact_store:
                return self.raw.artifact_store[key]

            result = func(*args, **kwargs)
            if isinstance(result, FilteredTensor):
                raise NotImplementedError(
                    "FilteredTensor caching is intentionally not supported."
                )
            if not isinstance(result, Tensor):
                return result

            self.raw.artifact_store[key] = result
            return result

        return wrapper

    def cache_some_other_call(self, func):  # left intentionally for extension
        raise NotImplementedError
