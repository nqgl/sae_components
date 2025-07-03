from __future__ import annotations

import sys

from collections.abc import Iterable, Iterator
from functools import cached_property
from types import EllipsisType
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    Generic,
    get_args,
    get_origin,
    get_type_hints,
    overload,
    Type,
    TypeVar,
)

import torch
from torch import Tensor
from typing_extensions import dataclass_transform

T = TypeVar("T")
from typing_extensions import Self


DictBatch_T = TypeVar("DictBatch_T", bound="DictBatch")


# --------------------------------------------------------------------------- #
#  Utility wrapper (kept exactly as in NiceBatch so existing pipes still work) #
# --------------------------------------------------------------------------- #
class NiceConvertedIter(Generic[DictBatch_T]):
    def __init__(self, iterable: Iterator[DictBatch_T]):
        self._iter = iterable

    def __iter__(self):  # pragma: no cover
        return self

    def __next__(self):  # pragma: no cover
        return next(self._iter)

    def __rshift__(  # Enables:   loader >> some_transform >> other_transform
        self,
        transform_gen: Callable[
            [Iterator[DictBatch_T]],
            Iterator[DictBatch_T] | Generator[DictBatch_T, None, None],
        ],
    ) -> "NiceConvertedIter[DictBatch_T]":
        return NiceConvertedIter(transform_gen(self._iter))

    def as_dataset(self) -> "NiceIterDataset[DictBatch_T]":
        return NiceIterDataset(self)


class NiceIterDataset(torch.utils.data.IterableDataset, Generic[DictBatch_T]):
    def __init__(self, iterable: Iterator[DictBatch_T]):
        self.iterable = iterable

    def __iter__(self) -> Iterator[DictBatch_T]:
        return self.iterable


# --------------------------------------------------------------------------- #
#                          The combined batch class                           #
# --------------------------------------------------------------------------- #
_PROTECTED = {
    "items",
    "values",
    "keys",
    "get",
    "setdefault",
    "pop",
    "update",
    "clear",
    "copy",
    "fromkeys",
    "popitem",
}


class DictBatch(dict):
    """
    A batch **is** a dict[str, torch.Tensor] *and* may carry extra attributes
    declared by subclasses via `OTHER_DATA_FIELDS` and `TENSOR_DATA_FIELDS`.

    Examples
    --------
    >>> class LMCombinedBatch(CombinedBatch):
    ...     OTHER_DATA_FIELDS = ("prompt", "doc_id")
    ...     TENSOR_DATA_FIELDS = ("input_ids", "attention_mask")
    ...
    >>> b = LMCombinedBatch(
    ...         {"input_ids": torch.ones(4, 128, dtype=torch.long),
    ...          "attention_mask": torch.ones(4, 128, dtype=torch.long)},
    ...         prompt="Once upon a time",
    ...         doc_id=42,
    ...     )
    >>> b.prompt, b["input_ids"].shape, b.input_ids.shape
    ('Once upon a time', torch.Size([4, 128]), torch.Size([4, 128]))
    >>> b2 = b[:2]
    >>> b2.prompt, len(b2)
    ('Once upon a time', 2)
    """

    # ------------------------------- metadata --------------------------------
    OTHER_DATA_FIELDS: ClassVar[tuple[str, ...]] = ()
    TENSOR_DATA_FIELDS: ClassVar[tuple[str, ...]] = ()

    # ------------------------------- ctor ------------------------------------
    def __init__(
        self,
        data: dict[str, Tensor] | None = None,
        /,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        data
            Mapping `str -> Tensor to initialise the dict body. If omitted,
            tensor kwargs are used instead.
        **kwargs
            *Either* tensor entries (for the dict) *or* values for
            `OTHER_DATA_FIELDS.
        """
        # Split kwargs into "extra" and "tensor" parts
        extra: dict[str, Any] = {}
        tensor_kwargs: dict[str, Tensor] = {}
        if (
            "data" in kwargs
            and data is None
            and isinstance(kwargs["data"], dict)
            and all(
                isinstance(k, str) and isinstance(v, Tensor)
                for k, v in kwargs["data"].items()
            )
        ):
            data = kwargs.pop("data")

        for k, v in kwargs.items():
            if k in self.OTHER_DATA_FIELDS:
                extra[k] = v
            else:
                tensor_kwargs[k] = v

        if data is not None:
            if tensor_kwargs:  # pragma: no cover
                raise ValueError(
                    "Provide tensors either via the data mapping OR as "
                    "keyword arguments, but not both."
                )
            self._check_contents(data)
            super().__init__(data)
        else:
            self._check_contents(tensor_kwargs)
            super().__init__(tensor_kwargs)

        # Check that all required TENSOR_DATA_FIELDS are present
        for field in self.TENSOR_DATA_FIELDS:
            if field not in self:
                raise ValueError(
                    f"Missing required tensor field '{field}' from TENSOR_DATA_FIELDS"
                )
            if not isinstance(self[field], Tensor):
                raise TypeError(
                    f"Field '{field}' from TENSOR_DATA_FIELDS must be a torch.Tensor, "
                    f"got {type(self[field])}"
                )

        # attach extras as attributes
        for k in self.OTHER_DATA_FIELDS:
            if k in extra:
                setattr(self, k, extra[k])
            else:
                # allow missing values, but keep attr for copy compat.
                setattr(self, k, None)

    # --------------------------- static helpers ------------------------------
    @property
    def data(self) -> dict[str, Tensor]:
        return self

    @staticmethod
    def _check_contents(d: dict[str, Tensor]):
        for k, v in d.items():
            if not isinstance(k, str):
                raise TypeError(f"Key {k!r} is not str")
            if k in _PROTECTED:
                raise KeyError(
                    f"Key {k!r} collides with protected dict method/attr name"
                )
            if not isinstance(v, Tensor):
                raise TypeError(f"Value for {k!r} is not a torch.Tensor")

    # ------------------------------ dunder -----------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.TENSOR_DATA_FIELDS:
            assert isinstance(value, Tensor)
            self[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """Allow accessing TENSOR_DATA_FIELDS as attributes."""
        if name in self.TENSOR_DATA_FIELDS:
            try:
                return self[name]
            except KeyError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @overload
    def __getitem__(self, key: str) -> Tensor: ...
    @overload
    def __getitem__(
        self,
        key: (
            int
            | slice
            | list[int]
            | Tensor
            | tuple[slice, ...]
            | tuple[EllipsisType, Tensor]
        ),
    ) -> Self: ...

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, str):
            return super().__getitem__(key)
        # fancy/regular indexing -> new batch
        return self.__class__.construct_with_other_data(
            {k: v[key] for k, v in self.items()}, self._get_other_dict()
        )

    # --------------------------- basic utilities -----------------------------
    @property
    def batch_size(self) -> int:
        size = next(iter(self.values())).shape[0]
        for v in self.values():
            assert v.shape[0] == size
        return size

    # ------- device helpers (borrowed from DictBatch) -------
    def to(self, *targets, **kwargs):
        return self.__class__.construct_with_other_data(
            {k: v.to(*targets, **kwargs) for k, v in self.items()},
            self._get_other_dict(),
        )

    def cuda(self, *args, **kwargs):
        return self.to("cuda", *args, **kwargs)

    # -------------------- cloning / contiguity -------------------------------
    def clone(self) -> Self:
        return self.__class__.construct_with_other_data(
            {k: v.clone() for k, v in self.items()}, self._get_other_dict()
        )

    def contiguous(self) -> Self:
        return self.__class__.construct_with_other_data(
            {k: v.contiguous() for k, v in self.items()}, self._get_other_dict()
        )

    # ----------------------- cat / stack helpers -----------------------------
    @classmethod
    def _validate_keysets(cls, batches: list[Self]):
        keys0 = batches[0].keys()
        if not all(b.keys() == keys0 for b in batches):
            raise ValueError("All batches must have identical tensor keys")

    @classmethod
    def cat_list(cls, batches: list[Self], dim: int = 0) -> Self:
        cls._validate_keysets(batches)
        return cls.construct_with_other_data(
            {k: torch.cat([b[k] for b in batches], dim=dim) for k in batches[0].keys()},
            cls._merge_other_data(batches),
        )

    @classmethod
    def stack_list(cls, batches: list[Self], dim: int = 0) -> Self:
        cls._validate_keysets(batches)
        return cls.construct_with_other_data(
            {
                k: torch.stack([b[k] for b in batches], dim=dim)
                for k in batches[0].keys()
            },
            cls._merge_other_data(batches),
        )

    # ---------------------- iterable conversion ------------------------------
    @classmethod
    def convert_iterable(
        cls, iterable: Iterable[dict[str, Tensor]]
    ) -> NiceConvertedIter[Self]:
        def gen():
            for d in iterable:
                yield cls(d)  # type: ignore[arg-type]

        return NiceConvertedIter(gen())

    # -------------------- helpers for extra data -----------------------------
    def _get_other_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.OTHER_DATA_FIELDS}

    @classmethod
    def construct_with_other_data(
        cls, data: dict[str, Tensor], other: dict[str, Any] | None = None
    ) -> Self:
        other = other or {}
        return cls(data, **other)

    # -- simple rule: enjoy equal extras or keep first; can customise in subclass
    @classmethod
    def _merge_other_data(cls, batches: list[Self]) -> dict[str, Any]:
        """Default strategy: require identical extras across all batches."""
        merged: dict[str, Any] = {}
        for field in cls.OTHER_DATA_FIELDS:
            values = [getattr(b, field) for b in batches]
            # If all equal → keep; else None (or override in subclass).
            merged[field] = values[0] if all(v == values[0] for v in values) else None
        return merged

    # --------------------------- len -----------------------------------------
    def __len__(self):
        l = len(next(iter(self.values())))
        for v in self.values():
            assert len(v) == l
        return l

    # ------------------------- representation --------------------------------
    def __repr__(self):  # pragma: no cover
        extra = ", ".join(f"{f}={getattr(self, f)!r}" for f in self.OTHER_DATA_FIELDS)
        return f"{self.__class__.__name__}({dict(self)}, {extra})"

    @staticmethod
    @dataclass_transform(kw_only_default=True)
    def auto_other_fields(cls: Type[T]) -> Type[T]:
        """
        Enhanced decorator that automatically populates OTHER_DATA_FIELDS and TENSOR_DATA_FIELDS from annotations.

        Features:
        - Handles forward references properly
        - Excludes ClassVar, property, and cached_property fields
        - Preserves manually defined OTHER_DATA_FIELDS and TENSOR_DATA_FIELDS if you want to add extras
        - Works with complex type annotations
        - Automatically detects Tensor-annotated fields for TENSOR_DATA_FIELDS
        """
        # Get type hints with forward reference resolution
        try:
            # This handles forward references better
            hints = get_type_hints(cls, include_extras=True)
        except Exception:
            # Fallback to raw annotations if get_type_hints fails
            hints = getattr(cls, "__annotations__", {})

        # Extract field names, separating tensor and other fields
        other_field_names = []
        tensor_field_names = []

        for name, hint in hints.items():
            # Skip if it's a method/property in the class
            if hasattr(cls, name):
                attr = getattr(cls, name)
                if isinstance(attr, (property, cached_property)):
                    continue
                if callable(attr) and not isinstance(attr, type):
                    continue

            # Skip ClassVar fields
            origin = get_origin(hint)
            if origin is ClassVar:
                continue

            # Skip if it starts with underscore (private)
            if name.startswith("_"):
                continue

            # Check if it's a Tensor type
            if hint is Tensor or (origin is not None and issubclass(origin, Tensor)):
                tensor_field_names.append(name)
            elif hint == "Tensor" or hint == "torch.Tensor":  # String annotations
                tensor_field_names.append(name)
            elif hasattr(hint, "array_type") and issubclass(hint.array_type, Tensor):
                # this is a jaxtyping annotated field (or something similar imitating that interface)
                # this doesn't make use of any of the nice features that jaxtyping provides,
                # so TODO add those features.

                tensor_field_names.append(name)
            else:
                other_field_names.append(name)

        # Get existing fields from parent classes
        parent_other_fields = ()
        parent_tensor_fields = ()
        for base in cls.__mro__[1:]:  # Skip the class itself
            if hasattr(base, "OTHER_DATA_FIELDS"):
                parent_other_fields = base.OTHER_DATA_FIELDS
            if hasattr(base, "TENSOR_DATA_FIELDS"):
                parent_tensor_fields = base.TENSOR_DATA_FIELDS
            if parent_other_fields or parent_tensor_fields:
                break

        # Handle OTHER_DATA_FIELDS
        if (
            hasattr(cls, "OTHER_DATA_FIELDS")
            and cls.OTHER_DATA_FIELDS != parent_other_fields
        ):
            # Merge manually defined fields with auto-detected ones
            manual_fields = tuple(
                f for f in cls.OTHER_DATA_FIELDS if f not in parent_other_fields
            )
            all_other_fields = (
                set(parent_other_fields) | set(other_field_names) | set(manual_fields)
            )
            cls.OTHER_DATA_FIELDS = tuple(all_other_fields)
        else:
            # Just combine parent fields with new fields
            cls.OTHER_DATA_FIELDS = parent_other_fields + tuple(other_field_names)

        # Handle TENSOR_DATA_FIELDS
        if (
            hasattr(cls, "TENSOR_DATA_FIELDS")
            and cls.TENSOR_DATA_FIELDS != parent_tensor_fields
        ):
            # Merge manually defined fields with auto-detected ones
            manual_tensor_fields = tuple(
                f for f in cls.TENSOR_DATA_FIELDS if f not in parent_tensor_fields
            )
            all_tensor_fields = (
                set(parent_tensor_fields)
                | set(tensor_field_names)
                | set(manual_tensor_fields)
            )
            cls.TENSOR_DATA_FIELDS = tuple(all_tensor_fields)
        else:
            # Just combine parent fields with new fields
            cls.TENSOR_DATA_FIELDS = parent_tensor_fields + tuple(tensor_field_names)

        return cls

    def float(self):
        return self.to(torch.float32)

    def items(self) -> Iterator[tuple[str, Tensor]]:
        return super().items()

    def gather(self, dim: int, indices: Tensor) -> Self:
        return self.construct_with_other_data(
            {k: v.gather(dim, indices) for k, v in self.items()}, self._get_other_dict()
        )


# Example usage:
if __name__ == "__main__":
    # Example 1: Using the decorator
    @DictBatch.auto_other_fields
    class MyBatch(DictBatch):
        input_ids: Tensor
        attention_mask: Tensor
        labels: Tensor
        prompt: str
        doc_id: int

    # This will automatically set:
    # TENSOR_DATA_FIELDS = ("input_ids", "attention_mask", "labels")
    # OTHER_DATA_FIELDS = ("prompt", "doc_id")

    # Create a batch
    batch = MyBatch(
        {
            "input_ids": torch.ones(4, 128, dtype=torch.long),
            "attention_mask": torch.ones(4, 128, dtype=torch.long),
            "labels": torch.ones(4, 128, dtype=torch.long),
        },
        prompt="Hello world",
        doc_id=42,
    )

    # Access tensors both ways
    print(batch["input_ids"].shape)  # torch.Size([4, 128])
    print(batch.input_ids.shape)  # torch.Size([4, 128])
    print(batch.attention_mask.shape)  # torch.Size([4, 128])

    # Access other fields
    print(batch.prompt)  # "Hello world"
    print(batch.doc_id)  # 42

    # This will raise an error - missing required tensor field
    try:
        bad_batch = MyBatch(
            {
                "input_ids": torch.ones(4, 128, dtype=torch.long),
                "attention_mask": torch.ones(4, 128, dtype=torch.long),
                # missing "labels"!
            },
            prompt="Hello world",
            doc_id=42,
        )
    except ValueError as e:
        print(
            f"Error: {e}"
        )  # Error: Missing required tensor field 'labels' from TENSOR_DATA_FIELDS

    def __or__(self, other: "DictBatch") -> "DictBatch":
        raise ValueError("ambiguous")

    def __and__(self, other: "DictBatch") -> "DictBatch":
        raise ValueError("ambiguous")
