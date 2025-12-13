from collections.abc import (
    Callable,
    Generator,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    ValuesView,
)
from functools import cached_property
from types import EllipsisType
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    cast,
    dataclass_transform,
    get_origin,
    get_type_hints,
    overload,
)

import torch
from attrs import define
from torch import Tensor

from saeco.misc.utils import assert_cast


def batch_size_targeter(batch_size: int):
    spares = []
    nspare = 0

    def yield_batches_and_return_spares[T: DictBatch](
        batch: T,
    ) -> Generator[T, None, T]:
        j = 0
        for i in range(0, batch.batch_size // batch_size * batch_size, batch_size):
            yield batch[i : i + batch_size]
            j += 1
        spare = batch[batch.batch_size // batch_size * batch_size :]
        return spare

    def transformed_gen[T: DictBatch](
        batch_gen: Iterable[T],
    ) -> Generator[T]:
        nonlocal spares, nspare
        for batch in batch_gen:
            spare = yield from yield_batches_and_return_spares(batch)

            if spare.batch_size > 0:
                spares.append(spare)
                nspare += spare.batch_size
                if nspare >= batch_size:
                    consolidated = batch.cat_list(spares, dim=0)
                    spare = yield from yield_batches_and_return_spares(consolidated)
                    spares = [spare]
                    nspare = spare.batch_size

    return transformed_gen


# --------------------------------------------------------------------------- #
#  Utility wrapper (kept exactly as in NiceBatch so existing pipes still work) #
# --------------------------------------------------------------------------- #
class NiceConvertedIter[DictBatch_T: "DictBatch"]:
    def __init__(self, iterable: Iterator[DictBatch_T]):
        self._iter = iterable

    def __iter__(self):  # pragma: no cover
        return self

    def __next__(self):  # pragma: no cover
        return next(self._iter)

    def __rshift__[
        Transformed_T: DictBatch
    ](  # Enables:   loader >> some_transform >> other_transform
        self,
        transform_gen: Callable[
            [Iterator[DictBatch_T]],
            Iterator[Transformed_T] | Generator[Transformed_T],
        ],
    ) -> "NiceConvertedIter[Transformed_T]":
        return NiceConvertedIter(transform_gen(self._iter))

    def as_dataset(self) -> "NiceIterDataset[DictBatch_T]":
        return NiceIterDataset(self)

    def as_dataloader(
        self, batch_size: int, num_workers: int = 4
    ) -> torch.utils.data.DataLoader[DictBatch_T]:
        return torch.utils.data.DataLoader(
            (self >> batch_size_targeter(batch_size)).as_dataset(),
            batch_size=None,
            num_workers=num_workers,
        )


class NiceIterDataset[DictBatch_T: "DictBatch"](torch.utils.data.IterableDataset):
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


class SkippedCalc:
    @overload
    @staticmethod
    def _skip_missing[R](
        func: Callable[[str, Tensor], R], pass_none_through: Literal[True]
    ) -> "Callable[[str, Tensor | None], R | None]": ...
    @overload
    @staticmethod
    def _skip_missing[R](
        func: Callable[[str, Tensor], R], pass_none_through: Literal[False] = False
    ) -> "Callable[[str, Tensor | None], R | SkippedCalc]": ...

    @staticmethod
    def _skip_missing[R](
        func: Callable[[str, Tensor], R], pass_none_through: bool = False
    ) -> "Callable[[str, Tensor | None], R | SkippedCalc | None]":
        def skipfunc(k: str, v: Tensor | None) -> R | SkippedCalc | None:
            if v is None:
                if pass_none_through:
                    return None
                return _SKIPPED_CALC
            return func(k, v)

        return skipfunc

    @staticmethod
    def _over_values[T: (Tensor, Tensor | None), R](
        func: Callable[[T], R],
    ) -> "Callable[[str, T], R]":
        def kvfunc(k: str, v: T) -> R:
            return func(v)

        return kvfunc


_SKIPPED_CALC = SkippedCalc()


@define
class _Default[T]:
    value: T | None = None
    func: Callable[[int], T] | None = None

    def render(self, batch_size: int) -> T | None:
        assert self.value is None or self.func is None
        if self.func is not None:
            return self.func(batch_size)
        return self.value


def dictbatch_field[T: Tensor = Tensor](*, factory: Callable[[int], T]) -> T:
    return cast(T, _Default(func=factory))


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
    FIELD_DEFAULTS: ClassVar[dict[str, _Default]] = {}
    OPTIONAL_TENSOR_FIELDS: ClassVar[tuple[str, ...]] = ()

    # ------------------------------- ctor ------------------------------------
    def __init__(
        self,
        data: dict[str, Tensor | None] | None = None,
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
            tensor_kwargs = data
        tensor_kwargs = {**self.FIELD_DEFAULTS, **tensor_kwargs}
        batch_sizes = [
            v.shape[0] for v in tensor_kwargs.values() if isinstance(v, Tensor)
        ]
        batch_size = batch_sizes[0]
        if not all(size == batch_size for size in batch_sizes):
            raise ValueError(
                f"All tensors must have the same batch size, got {batch_sizes}"
            )
        tensor_kwargs = {
            k: v if not isinstance(v, _Default) else v.render(batch_size)
            for k, v in tensor_kwargs.items()
        }
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
        for field in self.OPTIONAL_TENSOR_FIELDS:
            if field not in self:
                raise ValueError(
                    f"Missing required tensor field '{field}' from OPTIONAL_TENSOR_FIELDS"
                )
            if not isinstance(self[field], Tensor | None):
                raise TypeError(
                    f"Field '{field}' from OPTIONAL_TENSOR_FIELDS must be a torch.Tensor, "
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
    # @property
    # def data(self) -> dict[str, Tensor]:
    #     if "data" in self:
    #         return self["data"]
    #     elif "data" in self.__dict__:
    #         return self.__dict__["data"]

    #     warn(
    #         "DictBatch.data was used to access self, this backwards compatibility feature will be deprecated",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     return self  # just for backwards compatibility

    @classmethod
    def _check_contents(cls, d: dict[str, Tensor]):
        for k, v in d.items():
            if not isinstance(k, str):
                raise TypeError(f"Key {k!r} is not str")
            if k in _PROTECTED:
                raise KeyError(
                    f"Key {k!r} collides with protected dict method/attr name"
                )
            if not isinstance(v, Tensor):
                if k in cls.OPTIONAL_TENSOR_FIELDS or k not in cls.TENSOR_DATA_FIELDS:
                    if v is None:
                        continue
                    raise TypeError(
                        f"Value for {k!r} is not a torch.Tensor or None"
                        f"\n in class {cls}"
                    )
                raise TypeError(
                    f"Value for {k!r} is not a torch.Tensor {type(v)}\n in class {cls}"
                )

    # ------------------------------ dunder -----------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.TENSOR_DATA_FIELDS:
            assert isinstance(value, Tensor)
            self[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """Allow accessing TENSOR_DATA_FIELDS as attributes."""
        if name in self.TENSOR_DATA_FIELDS + self.OPTIONAL_TENSOR_FIELDS:
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

        def index_tensor(t: Tensor) -> Tensor:
            return t[key]

        return self.apply_func(index_tensor)

    def __setitem__(self, key: Any, value: Tensor) -> None:
        if key in self.TENSOR_DATA_FIELDS and not isinstance(value, Tensor):
            raise ValueError(f"Cannot set item {key} of type {type(key)}")
        super().__setitem__(key, value)

    # --------------------------- basic utilities -----------------------------
    def present_values(self) -> list[Tensor]:
        return [v for v in self.values() if v is not None]

    # ------- device helpers (borrowed from DictBatch) -------
    def to(self, *targets, **kwargs):
        def moveto(t: Tensor) -> Tensor:
            return t.to(*targets, **kwargs)

        return self.apply_func(moveto)

    def cuda(self, *args, **kwargs):
        return self.to("cuda", *args, **kwargs)

    # -------------------- cloning / contiguity -------------------------------
    def clone(self) -> Self:
        return self.apply_func(torch.clone)

    def contiguous(self) -> Self:
        def make_contiguous(t: Tensor) -> Tensor:
            return t.contiguous()

        return self.apply_func(make_contiguous)

    # ----------------------- cat / stack helpers -----------------------------
    @classmethod
    def _validate_keysets(cls, batches: list[Self], present_only: bool = False):
        def getkeys(b: Self):
            if present_only:
                return b.present_keys()
            return set(b.keys())

        keys0 = getkeys(batches[0])
        if not all(getkeys(b) == keys0 for b in batches):
            all_keys = [getkeys(b) for b in batches]
            min_keys = set.intersection(*all_keys)
            max_keys = set.union(*all_keys)

            batch_keys = "\n\t".join(
                f"{i}: {getkeys(b)}" for i, b in enumerate(batches)
            )
            raise ValueError(
                f"All batches must have identical tensor keys. Got: {batch_keys}"
                f"\nPresent in all batches: {min_keys}"
                f"\nkeys missing from some batches: {max_keys - min_keys}"
            )

    @classmethod
    def cat_list(cls, batches: list[Self], dim: int = 0) -> Self:
        cls._validate_keysets(batches, present_only=True)
        present = batches[0].present_keys()
        return cls.construct_with_other_data(
            {
                k: torch.cat([b[k] for b in batches] if k in present else None, dim=dim)
                for k in batches[0].keys()
            },
            cls._merge_other_data(batches),
        )

    @classmethod
    def stack_list(cls, batches: list[Self], dim: int = 0) -> Self:
        cls._validate_keysets(batches, present_only=True)
        present = batches[0].present_keys()
        return cls.construct_with_other_data(
            {
                k: torch.stack([b[k] for b in batches], dim=dim)
                if k in present
                else None
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
        cls, data: dict[str, Tensor | None], other: dict[str, Any] | None = None
    ) -> Self:
        other = other or {}
        return cls(data, **other)

    def _construct_with_copied_other_data(self, data: dict[str, Tensor | None]) -> Self:
        return self.construct_with_other_data(data, self._get_other_dict())

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
        return self.batch_size

    @property
    def batch_size(self) -> int:
        size = next(iter(self.present_values())).shape[0]
        for v in self.present_values():
            assert v.shape[0] == size
        return size

    # ------------------------- representation --------------------------------
    def __repr__(self):  # pragma: no cover
        extra = ", ".join(f"{f}={getattr(self, f)!r}" for f in self.OTHER_DATA_FIELDS)
        return f"{self.__class__.__name__}({dict(self)}, {extra})"

    @staticmethod
    @dataclass_transform(kw_only_default=True, field_specifiers=(dictbatch_field,))
    def auto_other_fields[T: DictBatch](cls: type[T]) -> type[T]:
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
        if not issubclass(cls, DictBatch):
            raise ValueError(f"Class {cls.__name__} is not a subclass of DictBatch")
        try:
            # This handles forward references better
            hints = get_type_hints(cls, include_extras=True)
        except Exception:
            # Fallback to raw annotations if get_type_hints fails
            hints = getattr(cls, "__annotations__", {})

        # Extract field names, separating tensor and other fields
        other_field_names = []
        tensor_field_names = []
        optional_tensor_field_names = []
        default_values = {}

        for name, hint in hints.items():
            # Skip if it's a method/property in the class
            is_tensor_field = False
            is_optional = False
            maybe_has_default = False
            default_value = None

            if hasattr(cls, name):
                attr = getattr(cls, name)
                if isinstance(attr, (property, cached_property)):
                    continue
                if callable(attr) and not isinstance(attr, type):
                    continue
                maybe_has_default = True
                default_value = attr

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
            elif hint == Tensor | None:
                optional_tensor_field_names.append(name)
                # tensor_field_names.append(name)?
            # elif issubclass(Tensor, hint):
            #     raise ValueError(
            #         f"Field {name} can be a tensor but failed to get unpacked"
            #     )
            else:
                other_field_names.append(name)
            if maybe_has_default:
                # assert isinstance(default_value, origin or hint), (
                #     default_value,
                #     origin,
                #     hint,
                # )

                default_values[name] = default_value

        # Get existing fields from parent classes
        all_other_fields = set(other_field_names)
        all_tensor_fields = set(tensor_field_names)
        all_optional_tensor_fields = set(optional_tensor_field_names)
        all_field_defaults = {}
        for base in cls.__mro__[1:]:  # Skip the class itself
            if hasattr(base, "OTHER_DATA_FIELDS"):
                base_fields = set(base.OTHER_DATA_FIELDS)
                all_other_fields |= base_fields
            if hasattr(base, "TENSOR_DATA_FIELDS"):
                base_fields = set(base.TENSOR_DATA_FIELDS)
                all_tensor_fields |= base_fields
            if hasattr(base, "OPTIONAL_TENSOR_FIELDS"):
                base_fields = set(base.OPTIONAL_TENSOR_FIELDS)
                all_optional_tensor_fields |= base_fields
            if hasattr(base, "FIELD_DEFAULTS"):
                base_defaults = base.FIELD_DEFAULTS
                all_field_defaults.update(base_defaults)
        if hasattr(cls, "OTHER_DATA_FIELDS"):
            all_other_fields |= set(cls.OTHER_DATA_FIELDS)

        if hasattr(cls, "TENSOR_DATA_FIELDS"):
            all_tensor_fields |= set(cls.TENSOR_DATA_FIELDS)
        if hasattr(cls, "OPTIONAL_TENSOR_FIELDS"):
            all_optional_tensor_fields |= set(cls.OPTIONAL_TENSOR_FIELDS)
        if hasattr(cls, "FIELD_DEFAULTS"):
            all_field_defaults.update(cls.FIELD_DEFAULTS)
        cls.OTHER_DATA_FIELDS = tuple(all_other_fields)
        cls.TENSOR_DATA_FIELDS = tuple(all_tensor_fields)
        cls.OPTIONAL_TENSOR_FIELDS = tuple(all_optional_tensor_fields)
        cls.FIELD_DEFAULTS = {**all_field_defaults, **default_values}

        return cls

    def float(self):
        return self.to(torch.float32)

    def items(self) -> ItemsView[str, Tensor | None]:
        return super().items()

    def values(self) -> ValuesView[Tensor | None]:
        return super().values()

    def keys(self) -> KeysView[str]:
        return super().keys()

    def present_keys(self) -> set[str]:
        return self._apply(
            lambda k, v: k, skip_none=True, return_set=True, takes_key=True
        )

    def gather(self, dim: int, indices: Tensor) -> Self:
        def gather(t: Tensor) -> Tensor:
            return t.gather(dim, indices)

        return self.apply_func(gather)

    def apply_func(self, func: Callable[[Tensor], Tensor]) -> Self:
        return self._construct_with_copied_other_data(self._apply(func, pass_none=True))

    @overload
    def _apply[T](
        self,
        func: Callable[[str, Tensor], T | SkippedCalc],
        *,
        takes_key: Literal[True],
        pass_none: Literal[True],
        return_set: Literal[True],
    ) -> set[T | None]: ...
    @overload
    def _apply[T](
        self,
        func: Callable[[str, Tensor], T | SkippedCalc],
        *,
        takes_key: Literal[True],
        skip_none: Literal[True],
        return_set: Literal[True],
    ) -> set[T]: ...

    @overload
    def _apply[T](
        self,
        func: Callable[[str, Tensor | None], T | SkippedCalc],
        *,
        takes_key: Literal[True],
        return_set: Literal[True],
    ) -> set[T]: ...

    @overload
    def _apply[T](
        self,
        func: Callable[[Tensor], T | SkippedCalc],
        *,
        pass_none: Literal[True],
        return_set: Literal[True],
    ) -> set[T | None]: ...
    @overload
    def _apply[T](
        self,
        func: Callable[[Tensor], T | SkippedCalc],
        *,
        skip_none: Literal[True],
        return_set: Literal[True],
    ) -> set[T]: ...

    @overload
    def _apply[T](
        self,
        func: Callable[[Tensor | None], T | SkippedCalc],
        *,
        return_set: Literal[True],
    ) -> set[T]: ...

    @overload
    def _apply[T](
        self,
        func: Callable[[str, Tensor], T | SkippedCalc],
        *,
        takes_key: Literal[True],
        pass_none: Literal[True],
    ) -> dict[str, T | None]: ...
    @overload
    def _apply[T](
        self,
        func: Callable[[str, Tensor], T | SkippedCalc],
        *,
        takes_key: Literal[True],
        skip_none: Literal[True],
    ) -> dict[str, T]: ...

    @overload
    def _apply[T](
        self,
        func: Callable[[str, Tensor | None], T | SkippedCalc],
        *,
        takes_key: Literal[True],
    ) -> dict[str, T]: ...

    @overload
    def _apply[T](
        self,
        func: Callable[[Tensor], T | SkippedCalc],
        *,
        pass_none: Literal[True],
    ) -> dict[str, T | None]: ...
    @overload
    def _apply[T](
        self,
        func: Callable[[Tensor], T | SkippedCalc],
        *,
        skip_none: Literal[True],
    ) -> dict[str, T]: ...

    @overload
    def _apply[T](
        self,
        func: Callable[[Tensor | None], T | SkippedCalc],
    ) -> dict[str, T]: ...

    def _apply[T](
        self,
        func: Callable[[str, Tensor | None], T | SkippedCalc]
        | Callable[[Tensor | None], T | SkippedCalc]
        | Callable[[str, Tensor], T | SkippedCalc]
        | Callable[[Tensor], T | SkippedCalc],
        *,
        pass_none: bool = False,
        skip_none: bool = False,
        takes_key: bool = False,
        return_set: bool = False,
    ) -> dict[str, T | None] | dict[str, T] | set[T] | set[T | None]:
        if not takes_key:
            func = SkippedCalc._over_values(
                cast(
                    Callable[[Tensor | None], T | SkippedCalc]
                    | Callable[[Tensor], T | SkippedCalc],
                    func,
                )
            )
        else:
            func = cast(
                Callable[[str, Tensor | None], T | SkippedCalc]
                | Callable[[str, Tensor], T | SkippedCalc],
                func,
            )
        assert not (pass_none and skip_none)
        maybe_returns_none_func: (
            Callable[[str, Tensor | None], T | SkippedCalc | None]
            | Callable[[str, Tensor], T | SkippedCalc | None]
        ) = func
        if pass_none:
            maybe_returns_none_func = SkippedCalc._skip_missing(
                func, pass_none_through=False
            )
        elif skip_none:
            f = cast(
                Callable[[str, Tensor], T | SkippedCalc],
                func,
            )

            maybe_returns_none_func = SkippedCalc._skip_missing(
                f,
                pass_none_through=True,
            )
        else:
            maybe_returns_none_func = cast(
                (Callable[[str, Tensor | None], T | SkippedCalc | None]), func
            )
        if return_set:
            return {
                result
                for k, v in self.items()
                if not isinstance(result := maybe_returns_none_func(k, v), SkippedCalc)
            }

        return {
            k: result
            for k, v in self.items()
            if not isinstance(result := maybe_returns_none_func(k, v), SkippedCalc)
        }

    def reshape(self, *shape: int) -> Self:
        return self.apply_func(lambda x: x.reshape(*shape))

    def set_split(
        self, sel: tuple[str, ...] | set[str] | list[str]
    ) -> "SplitDictBatch[Self]":
        if not isinstance(sel, set):
            sel = set(sel)
        keys = set(self.keys())
        if sel - keys:
            raise ValueError(f"Keys {sel - keys} not found in {keys}")
        a_keys = sel
        b_keys = keys - sel
        a = DictBatch({k: v for k, v in self.items() if k in a_keys})
        b = DictBatch({k: v for k, v in self.items() if k in b_keys})
        return SplitDictBatch(a, b, self._get_other_dict(), self.__class__)

    def index_subset(
        self,
        mask: Tensor,
        include: list[str] | set[str] | None = None,
        exclude: list[str] | set[str] | None = None,
    ) -> Self:
        if (include is None) == (exclude is None):
            raise ValueError("Either include or exclude must be provided, but not both")
        if isinstance(include, list):
            include = set(include)
        if isinstance(exclude, list):
            exclude = set(exclude)
        if include is None:
            include = set(self.keys()) - exclude

        split = self.set_split(include)
        split.a = split.a[mask]
        return split.recombine()

    def split(
        self,
        split_size: int | list[int],
        dim: int = 0,
    ) -> list[Self]:
        assert dim == 0
        splitself = self._apply(lambda x: x.split(split_size, dim=dim), skip_none=True)
        # splitself = {k: v.split(split_size, dim=dim) for k, v in self.items()}
        l0 = next(iter(splitself.values()))
        assert all(len(v) == len(l0) for v in splitself.values())

        return [
            self._construct_with_copied_other_data(
                {
                    **{
                        k: splitself[k][i] if k in splitself else None
                        for k in self.keys()
                    }
                },
            )
            for i in range(len(l0))
        ]

    def einops_rearrange(self, pattern: str, **kwargs):
        import einops

        def rearrange(t: Tensor) -> Tensor:
            return einops.rearrange(t, pattern, **kwargs)

        return self.apply_func(rearrange)

    def get_unanimous_result[T](self, func: Callable[[Tensor], T]) -> T:
        results = self._apply(func, return_set=True, skip_none=True)
        if len(results) != 1:
            raise ValueError(f"Expected 1 result, got {len(results)}")
        return results.pop()

    @property
    def device(self) -> torch.device:
        return self.get_unanimous_result(lambda x: x.device)

    @property
    def is_sparse(self) -> bool:
        return any(v.is_sparse for v in self.present_values())

    @property
    def is_coalesced(self) -> bool:
        return all((not v.is_sparse) or v.is_coalesced() for v in self.present_values())

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return self._apply(lambda x: x.shape, skip_none=True)

    @property
    def shape(self) -> "DictBatchShape":
        return DictBatchShape(batch_size=self.batch_size, shapes=self.shapes)

    def updated_with(self, **kwargs: Tensor) -> Self:
        return self.construct_with_other_data(
            {**self, **kwargs}, self._get_other_dict()
        )

    @classmethod
    def cast_convert(
        cls,
        batch: "DictBatch",
        **kwargs,
    ) -> Self:
        extra_data = {
            k: v
            for k, v in kwargs.items()
            if k in cls.TENSOR_DATA_FIELDS + cls.OPTIONAL_TENSOR_FIELDS
        }
        extra_other_data = {
            k: v for k, v in kwargs.items() if k in cls.OTHER_DATA_FIELDS
        }
        if isinstance(batch, cls) and type(batch) is cls:
            return batch
        return cls.construct_with_other_data(
            {**batch, **extra_data}, {**batch._get_other_dict(), **extra_other_data}
        )


@define
class DictBatchShape:
    batch_size: int
    shapes: dict[str, tuple[int, ...]]

    @overload
    def __getitem__(self, key: str) -> tuple[int, ...]: ...
    @overload
    def __getitem__(self, key: Literal[0]) -> int: ...
    @overload
    def __getitem__(self, key: int) -> dict[str, int]: ...
    def __getitem__(self, key: str | int) -> tuple[int, ...] | int | dict[str, int]:
        if key == 0:
            return self.batch_size
        if isinstance(key, str):
            return self.shapes[key]
        assert isinstance(key, int)
        return {k: v[key] for k, v in self.shapes.items() if len(v) > key}


@define
class DictBatchShapes:
    batch_sizes: tuple[int, ...]
    shapes: dict[str, tuple[int, ...]]

    @overload
    def __getitem__(self, key: str) -> tuple[int, ...]: ...
    @overload
    def __getitem__(self, key: Literal[0]) -> int: ...
    @overload
    def __getitem__(self, key: int) -> dict[str, int]: ...
    def __getitem__(self, key: str | int) -> tuple[int, ...] | int | dict[str, int]:
        if isinstance(key, int) and key < len(self.batch_sizes):
            return self.batch_sizes[key]
        if isinstance(key, str):
            return self.shapes[key]
        assert isinstance(key, int)
        return {k: v[key] for k, v in self.shapes.items() if len(v) > key}

    def __radd__(self, other) -> Self:
        if isinstance(other, (list, tuple)):
            other_tup = tuple(assert_cast(int, o) for o in other)
            return self.__class__(
                batch_sizes=other_tup + self.batch_sizes, shapes=self.shapes
            )
        raise ValueError(f"Cannot add {type(other)} to {type(self)}")


@define
class SplitDictBatch[T: DictBatch]:
    a: DictBatch
    b: DictBatch
    other_data: dict[str, Any]
    cls: type[T]

    def recombine(self) -> T:
        return self.cls.construct_with_other_data({**self.a, **self.b}, self.other_data)


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

    def ones_maker(batch_size: int):
        return torch.ones(batch_size, 128, dtype=torch.long)

    @DictBatch.auto_other_fields
    class WithOptionalTensor(DictBatch):
        a: Tensor = dictbatch_field(factory=ones_maker)
        b: Tensor | None
        c: Tensor | None = None

    batch = WithOptionalTensor(
        a=torch.ones(4, 128, dtype=torch.long),
        b=torch.ones(4, 128, dtype=torch.long),
        c=torch.ones(4, 128, dtype=torch.long),
    )
    print(batch)
    batch = WithOptionalTensor(
        a=torch.ones(4, 128, dtype=torch.long),
        b=torch.ones(4, 128, dtype=torch.long),
    )
    batch.updated_with(a=torch.ones(4, 128, dtype=torch.long))

    batch = WithOptionalTensor(
        b=torch.ones(4, 128, dtype=torch.long),
        c=torch.ones(4, 128, dtype=torch.long),
    )
    print(batch)
    batch = WithOptionalTensor(
        a=torch.ones(4, 128, dtype=torch.long),
        b=None,
    )
    # should error:
    batch = WithOptionalTensor(
        a=torch.ones(4, 128, dtype=torch.long),
        # b=None,
    )

    # should error:
    batch = WithOptionalTensor(
        a=None,
        b=None,
    )
