from collections.abc import (
    Callable,
    Generator,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    ValuesView,
)
from functools import cached_property
from pathlib import Path
from types import EllipsisType, UnionType
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import torch
from attrs import define
from paramsight.type_utils import get_args_robust, get_origin_robust
from torch import Tensor

from saeco.data.storage.compressed_safetensors import load_file_with_metadata
from saeco.misc.utils import assert_cast, chill_issubclass


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

type _DTDataT = "dict[str, _DTDataT ] |  Tensor | DictBatch | None"


def _flatten(
    d: "Mapping[str, _DTDataT] | DictBatch",
) -> Mapping[str, Tensor | None]:
    """
    Flatten a nested dictionary or DictBatch into a flat dictionary of
    str -> (Tensor | None)
    """
    already_flat = {k: v for k, v in d.items() if not isinstance(v, dict)}
    flattened = {
        f"{k}.{k1}": v1
        for k, v in d.items()
        if isinstance(v, dict)
        for k1, v1 in _flatten(v).items()
    }
    return {**already_flat, **flattened}


def _unflatten_once_strict[T](d: dict[str, T]) -> dict[str, dict[str, T]]:
    """
        transforms dictionary whose keys have a "." key-path-separator in the key
        to nested dictionaries with the prefix as a key tot he
        creates one layer of nesting for

        takes a dictionary of str -> T
        and transforms keys of the form {"k0.k1...": v1, "k0.k2...": v2}
    Q    into {"k0": {"k1...": v1, "k2...": v2}}
    """
    d_out = {}
    items = []
    for k, v in d.items():
        if "." not in k or "." in (k[0], k[-1]) or ".." in k:
            d_out[k] = v
        else:
            items.append((k, v))

        # if "." not in k:
        #     raise ValueError(f"Key {k} does not have a '.' separator")
        # if "." in (k[0], k[-1]):
        #     raise ValueError(f"Key {k} has a '.' separator at the beginning or end")
        # if ".." in k:
        #     raise ValueError(f"Key {k} has a '..' sequence")

    for k, v in items:
        k0, *k_ = k.split(".")
        k_rest = ".".join(k_)
        assert len(k_rest) > 0
        if k0 not in d_out:
            d_out[k0] = {}
        d_out[k0][k_rest] = v
    assert "" not in d_out
    return d_out


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


_NONE_TYPE = type(None)


def _hint_allows_none(hint: Any) -> bool:
    if hint is Any:
        return True
    if hint is _NONE_TYPE:
        return True
    return _NONE_TYPE in get_args(hint)


def _hint_allows_tensor(hint: Any) -> bool:
    if hint is Tensor or hint == "Tensor" or hint == "torch.Tensor":
        return True

    if hasattr(hint, "array_type"):
        try:
            return issubclass(hint.array_type, Tensor)
        except TypeError:
            return False

    origin = get_origin(hint)
    if origin is not None:
        try:
            if isinstance(origin, type) and issubclass(origin, Tensor):
                return True
        except TypeError:
            pass

    args = get_args(hint)
    if args:
        return any(_hint_allows_tensor(a) for a in args if a is not _NONE_TYPE)

    return False


def _hint_allows_dictbatch(hint: Any) -> bool:
    if hint is Any:
        raise ValueError("Any is not a valid hint for a DictBatch")
    if hint in {"DictBatch", "saeco.data.dict_batch.dict_batch.DictBatch"}:
        return True

    if isinstance(hint, type):
        try:
            return issubclass(hint, DictBatch)
        except (NameError, TypeError):
            return False
    origin = get_origin(hint)
    if origin and isinstance(origin, type) and issubclass(origin, DictBatch):
        return True
    args = get_args(hint)
    if args:
        return any(_hint_allows_dictbatch(a) for a in args if a is not _NONE_TYPE)

    return False


def _get_dictbatch_constructor_type(hint) -> type["DictBatch"]:
    if chill_issubclass(hint, DictBatch):
        return hint
    origin = get_origin_robust(hint)
    if origin is UnionType:
        args = get_args_robust(hint)
        (a,) = [a for a in args if a is not None]
        assert issubclass(a, DictBatch)
        return a
    raise ValueError(f"Invalid hint: {hint!r}")


def _is_tensor_field_hint(hint: Any) -> bool:
    if _hint_allows_tensor(hint) or _hint_allows_dictbatch(hint):
        args = get_args(hint)
        if args:
            non_none_args = [a for a in args if a is not _NONE_TYPE]
            return bool(non_none_args) and all(
                _is_tensor_field_hint(a) for a in non_none_args
            )
        return True

    return False


def _validate_tensor_field_value(field: str, hint: Any, value: Any) -> None:
    if value is None:
        if not _hint_allows_none(hint):
            raise TypeError(f"Field {field!r} cannot be None (hint: {hint!r})")
        return

    if isinstance(value, Tensor):
        if hint is not Any and not _hint_allows_tensor(hint):
            raise TypeError(
                f"Field {field!r} expects a DictBatch-like value (hint: {hint!r}), "
                "got Tensor"
            )
        return

    if isinstance(value, DictBatch):
        if hint is not Any and not _hint_allows_dictbatch(hint):
            raise TypeError(
                f"Field {field!r} expects a Tensor-like value (hint: {hint!r}), "
                f"got {type(value)}"
            )

        args = get_args(hint)
        if isinstance(hint, type):
            allowed = (hint,)
        elif args:
            allowed = tuple(
                a for a in args if isinstance(a, type) and issubclass(a, DictBatch)
            )
        else:
            allowed = ()

        if allowed and not any(isinstance(value, a) for a in allowed):
            raise TypeError(f"Field {field!r} expects {allowed}, got {type(value)}")
        return

    raise TypeError(
        f"Field {field!r} must be a torch.Tensor, a DictBatch, or None; "
        f"got {type(value)}"
    )


type TensorFieldTypes = Tensor | DictBatch | None


class DictBatch(dict):
    """
    A batch **is** a dict[str, torch.Tensor | DictBatch] *and* may carry extra
    attributes
    declared by subclasses via `OTHER_DATA_FIELDS` and `TENSOR_DATA_FIELDS`.

    Examples
    --------
    >>> class LMCombinedBatch(DictBatch):
    ...     OTHER_DATA_FIELDS = {"prompt": str, "doc_id": int}
    ...     TENSOR_DATA_FIELDS = {"input_ids": Tensor, "attention_mask": Tensor}
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
    OTHER_DATA_FIELDS: ClassVar[dict[str, Any]] = {}
    TENSOR_DATA_FIELDS: ClassVar[dict[str, Any]] = {}
    FIELD_DEFAULTS: ClassVar[dict[str, Tensor | None | _Default]] = {}
    OTHER_FIELD_DEFAULTS: ClassVar[dict[str, Any]] = {}

    # ------------------------------- ctor ------------------------------------
    def __init__(
        self,
        data: dict[str, "Tensor | None | DictBatch"] | None = None,
        /,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        data
            Mapping `str -> (Tensor | DictBatch)` to initialise the dict body.
            If omitted, tensor kwargs are used instead.
        **kwargs
            *Either* tensor entries (for the dict) *or* values for
            `OTHER_DATA_FIELDS.
        """
        # Split kwargs into "extra" and "tensor" parts
        extra: dict[str, Any] = {}
        tensor_kwargs: dict[str, Tensor | None] = {}
        if (
            "data" in kwargs
            and data is None
            and isinstance(kwargs["data"], dict)
            and all(
                isinstance(k, str)
                and (isinstance(v, Tensor) or isinstance(v, DictBatch) or v is None)
                for k, v in kwargs["data"].items()
            )
        ):
            data = kwargs.pop("data")  # type: ignore[assignment]

        for k, v in kwargs.items():
            if k in self.OTHER_DATA_FIELDS:
                extra[k] = v
            else:
                tensor_kwargs[k] = v  # type: ignore[assignment]

        if data is not None:
            if tensor_kwargs:  # pragma: no cover
                raise ValueError(
                    "Provide tensors either via the data mapping OR as "
                    "keyword arguments, but not both."
                )
            tensor_kwargs = data
        tensor_kwargs = {**self.FIELD_DEFAULTS, **tensor_kwargs}
        batch_sizes: list[int] = []
        for v in tensor_kwargs.values():
            if isinstance(v, Tensor):
                batch_sizes.append(v.shape[0])
            elif isinstance(v, DictBatch):
                batch_sizes.append(v.batch_size)

        if not batch_sizes:
            raise ValueError(
                "Cannot infer batch size: no tensors (or nested DictBatches) provided."
            )

        batch_size = batch_sizes[0]
        if not all(size == batch_size for size in batch_sizes):
            raise ValueError(
                f"All tensors must have the same batch size, got {batch_sizes}"
            )
        tensor_kwargs = {
            k: v if not isinstance(v, _Default) else v.render(batch_size)
            for k, v in tensor_kwargs.items()
        }
        post_sizes: list[int] = []
        for v in tensor_kwargs.values():
            if isinstance(v, Tensor):
                post_sizes.append(v.shape[0])
            elif isinstance(v, DictBatch):
                post_sizes.append(v.batch_size)
            elif v is None:
                continue
            else:  # pragma: no cover
                raise TypeError(
                    f"Unexpected value type in batch: {type(v)} for value {v!r}"
                )
        if post_sizes and not all(size == batch_size for size in post_sizes):
            raise ValueError(
                f"All tensors must have the same batch size, got {post_sizes}"
            )
        self._check_contents(tensor_kwargs)
        super().__init__(tensor_kwargs)

        # Check that all declared tensor fields are present and match optionality.
        for field, hint in self.TENSOR_DATA_FIELDS.items():
            if field not in self:
                raise ValueError(
                    f"Missing required tensor field '{field}' from TENSOR_DATA_FIELDS"
                )
            _validate_tensor_field_value(field, hint, self[field])

        # attach extras as attributes
        for k in self.OTHER_DATA_FIELDS:
            if k in extra:
                setattr(self, k, extra[k])
            elif k in self.OTHER_FIELD_DEFAULTS:
                setattr(self, k, self.OTHER_FIELD_DEFAULTS[k])
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
    #         "DictBatch.data was used to access self, this backwards compatibility "
    #         "feature will be deprecated",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     return self  # just for backwards compatibility

    @classmethod
    def _check_contents(cls, d: dict[str, Any]):
        for k, v in d.items():
            if not isinstance(k, str):
                raise TypeError(f"Key {k!r} is not str")
            if k in _PROTECTED:
                raise KeyError(
                    f"Key {k!r} collides with protected dict method/attr name"
                )
            if k in cls.TENSOR_DATA_FIELDS:
                _validate_tensor_field_value(k, cls.TENSOR_DATA_FIELDS[k], v)
                continue

            if v is None:
                continue
            if isinstance(v, Tensor) or isinstance(v, DictBatch):
                continue
            raise TypeError(
                f"Value for {k!r} is not a torch.Tensor, DictBatch, or None "
                f"({type(v)})\n"
                f"in class {cls}"
            )

    # ------------------------------ dunder -----------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.TENSOR_DATA_FIELDS:
            self[name] = value
        else:
            super().__setattr__(name, value)

    # def __getattribute__(self, name: str) -> Any:
    #     """Allow accessing TENSOR_DATA_FIELDS as attributes."""
    #     if name != "__class__" and name in self.__class__.TENSOR_DATA_FIELDS:
    #         return self[name]
    #     return super().__getattribute__(name)

    def __getattr__(self, name: str) -> Any:
        """Allow accessing TENSOR_DATA_FIELDS as attributes."""
        if name in self.TENSOR_DATA_FIELDS:
            try:
                return self[name]
            except KeyError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                ) from None
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @overload
    def __getitem__(self, key: str) -> Any: ...
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

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str) and key in self.TENSOR_DATA_FIELDS:
            _validate_tensor_field_value(key, self.TENSOR_DATA_FIELDS[key], value)
        super().__setitem__(key, value)

    # --------------------------- basic utilities -----------------------------
    def iter_tensors(self) -> Iterator[Tensor]:
        for v in self.values():
            if v is None:
                continue
            if isinstance(v, Tensor):
                yield v
                continue
            if isinstance(v, DictBatch):
                yield from v.iter_tensors()
                continue
            raise TypeError(f"Unexpected value type in DictBatch: {type(v)}")

    def present_values(self) -> list[Tensor]:
        return list(self.iter_tensors())

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

        # Recursively validate any nested DictBatch values for the selected keys.
        for key in keys0:
            values = [b[key] for b in batches]
            non_none_values = [v for v in values if v is not None]

            if not any(isinstance(v, DictBatch) for v in non_none_values):
                continue

            if not all(isinstance(v, DictBatch) for v in non_none_values):
                raise TypeError(
                    f"Key {key!r} mixes DictBatch and non-DictBatch values: "
                    f"{[type(v) for v in non_none_values]}"
                )

            if len(non_none_values) != len(values):
                # Optional nested batch missing in some inputs; handled by
                # present_only validation.
                continue

            child0 = cast(DictBatch, non_none_values[0])
            child_type = type(child0)
            if not all(type(v) is child_type for v in non_none_values):
                raise TypeError(
                    f"Key {key!r} has inconsistent nested batch types: "
                    f"{[type(v) for v in non_none_values]}"
                )

            child_type._validate_keysets(
                cast(list[DictBatch], non_none_values), present_only=present_only
            )

    @classmethod
    def cat_list(cls, batches: list[Self], dim: int = 0) -> Self:
        cls._validate_keysets(batches)
        cls._validate_keysets(batches, present_only=True)
        present = batches[0].present_keys()
        out: dict[str, Any] = {}
        for k in batches[0].keys():
            if k not in present:
                out[k] = None
                continue

            values = [b[k] for b in batches]
            v0 = values[0]
            if isinstance(v0, Tensor):
                out[k] = torch.cat(cast(list[Tensor], values), dim=dim)
                continue
            if isinstance(v0, DictBatch):
                child_type = type(v0)
                if not all(
                    isinstance(v, DictBatch) and type(v) is child_type for v in values
                ):
                    raise TypeError(
                        f"Key {k!r} has inconsistent nested batch types: "
                        f"{[type(v) for v in values]}"
                    )
                out[k] = child_type.cat_list(cast(list[DictBatch], values), dim=dim)
                continue

            raise TypeError(f"Unexpected value type for key {k!r}: {type(v0)}")

        return cls.construct_with_other_data(out, cls._merge_other_data(batches))

    @classmethod
    def stack_list(cls, batches: list[Self], dim: int = 0) -> Self:
        cls._validate_keysets(batches, present_only=True)
        cls._validate_keysets(batches)
        present = batches[0].present_keys()
        out: dict[str, Any] = {}
        for k in batches[0].keys():
            if k not in present:
                out[k] = None
                continue

            values = [b[k] for b in batches]
            v0 = values[0]
            if isinstance(v0, Tensor):
                out[k] = torch.stack(cast(list[Tensor], values), dim=dim)
                continue
            if isinstance(v0, DictBatch):
                child_type = type(v0)
                if not all(
                    isinstance(v, DictBatch) and type(v) is child_type for v in values
                ):
                    raise TypeError(
                        f"Key {k!r} has inconsistent nested batch types: "
                        f"{[type(v) for v in values]}"
                    )
                out[k] = child_type.stack_list(cast(list[DictBatch], values), dim=dim)
                continue

            raise TypeError(f"Unexpected value type for key {k!r}: {type(v0)}")

        return cls.construct_with_other_data(out, cls._merge_other_data(batches))

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
        cls, data: dict[str, Any], other: dict[str, Any] | None = None
    ) -> Self:
        other = other or {}
        return cls(data, **other)

    def _construct_with_copied_other_data(self, data: dict[str, Any]) -> Self:
        return self.cast_convert(data, **self._get_other_dict())

    # -- simple rule: enjoy equal extras or keep first; can customise in subclass
    @classmethod
    def _merge_other_data(
        cls, batches: list[Self], strict: bool = True
    ) -> dict[str, Any]:
        """Default strategy: require identical extras across all batches."""
        merged: dict[str, Any] = {}
        for field in cls.OTHER_DATA_FIELDS:
            values = [getattr(b, field) for b in batches]
            # If all equal → keep; else None (or override in subclass).
            if all(v == values[0] for v in values):
                merged[field] = values[0]
            else:
                if strict:
                    raise ValueError(
                        f"Other data field {field} has inconsistent values: {values}"
                    )
                merged[field] = None
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
        # return f"{self.__class__.__name__}({dict(self)}, {extra})"

        """show shapes and types, not values"""
        reprs = {
            k: repr(v.shape) if isinstance(v, Tensor) else repr(v)
            for k, v in self.items()
        }
        return f"{self.__class__.__name__}({reprs}, {extra})"

    @staticmethod
    @dataclass_transform(kw_only_default=True, field_specifiers=(dictbatch_field,))
    def auto_other_fields[T: DictBatch](cls: type[T]) -> type[T]:
        """
        Enhanced decorator that automatically populates OTHER_DATA_FIELDS and
        TENSOR_DATA_FIELDS from annotations.

        Features:
        - Handles forward references properly
        - Excludes ClassVar, property, and cached_property fields
        - Preserves manually defined OTHER_DATA_FIELDS and TENSOR_DATA_FIELDS if you
          want to add extras
        - Works with complex type annotations
        - Automatically detects Tensor-annotated fields for TENSOR_DATA_FIELDS
        """
        # Get type hints with forward reference resolution
        if not issubclass(cls, DictBatch):
            raise ValueError(f"Class {cls.__name__} is not a subclass of DictBatch")
        # try:
        # This handles forward references better
        hints = get_type_hints(cls, include_extras=False)
        # except Exception:
        #     # Fallback to raw annotations if get_type_hints fails
        #     hints = getattr(cls, "__annotations__", {})

        tensor_field_hints: dict[str, Any] = {}
        other_field_hints: dict[str, Any] = {}
        tensor_defaults: dict[str, Any] = {}
        other_defaults: dict[str, Any] = {}

        for name, hint in hints.items():
            maybe_has_default = False
            default_value = None

            # Skip if it's a method/property in the class
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

            if _is_tensor_field_hint(hint):
                tensor_field_hints[name] = hint
                if maybe_has_default:
                    tensor_defaults[name] = default_value
            else:
                other_field_hints[name] = hint
                if maybe_has_default:
                    other_defaults[name] = default_value

        def normalize_field_map(
            v: object | None, *, default_type: object
        ) -> dict[str, Any]:
            if v is None:
                return {}
            if isinstance(v, Mapping):
                return dict(v)
            if isinstance(v, (tuple, list, set)):
                return {assert_cast(str, k): default_type for k in v}
            raise TypeError(f"Unsupported field spec: {type(v)}")

        merged_other: dict[str, Any] = {}
        merged_tensor: dict[str, Any] = {}
        merged_tensor_defaults: dict[str, Any] = {}
        merged_other_defaults: dict[str, Any] = {}

        # Merge base classes first (so subclass annotations override).
        for base in reversed(cls.__mro__[1:]):  # Skip the class itself
            merged_other.update(
                normalize_field_map(
                    getattr(base, "OTHER_DATA_FIELDS", None), default_type=Any
                )
            )
            merged_tensor.update(
                normalize_field_map(
                    getattr(base, "TENSOR_DATA_FIELDS", None), default_type=Tensor
                )
            )

            merged_tensor_defaults.update(getattr(base, "FIELD_DEFAULTS", {}) or {})
            merged_other_defaults.update(
                getattr(base, "OTHER_FIELD_DEFAULTS", {}) or {}
            )

            # Backwards-compat: older subclasses may still declare
            # OPTIONAL_TENSOR_FIELDS.
            for field in getattr(base, "OPTIONAL_TENSOR_FIELDS", ()):
                merged_tensor[field] = Tensor | None

        # Preserve any manual additions on the class.
        merged_other.update(
            normalize_field_map(
                getattr(cls, "OTHER_DATA_FIELDS", None), default_type=Any
            )
        )
        merged_tensor.update(
            normalize_field_map(
                getattr(cls, "TENSOR_DATA_FIELDS", None), default_type=Tensor
            )
        )
        merged_tensor_defaults.update(getattr(cls, "FIELD_DEFAULTS", {}) or {})
        merged_other_defaults.update(getattr(cls, "OTHER_FIELD_DEFAULTS", {}) or {})

        for field in getattr(cls, "OPTIONAL_TENSOR_FIELDS", ()):
            merged_tensor[field] = Tensor | None

        # Apply current class annotations (overrides base/manual types).
        merged_other.update(other_field_hints)
        merged_tensor.update(tensor_field_hints)
        merged_tensor_defaults.update(tensor_defaults)
        merged_other_defaults.update(other_defaults)

        cls.OTHER_DATA_FIELDS = merged_other
        cls.TENSOR_DATA_FIELDS = merged_tensor
        cls.FIELD_DEFAULTS = merged_tensor_defaults
        cls.OTHER_FIELD_DEFAULTS = merged_other_defaults

        return cls

    def float(self):
        return self.to(torch.float32)

    def items(self) -> ItemsView[str, TensorFieldTypes]:
        return super().items()

    def values(self) -> ValuesView[TensorFieldTypes]:
        return super().values()

    def keys(self) -> KeysView[str]:
        return super().keys()

    def present_keys(self) -> set[str]:
        return {k for k, v in self.items() if v is not None}

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
                func, pass_none_through=True
            )
        elif skip_none:
            f = cast(
                Callable[[str, Tensor], T | SkippedCalc],
                func,
            )

            maybe_returns_none_func = SkippedCalc._skip_missing(
                f,
                pass_none_through=False,
            )
        else:
            maybe_returns_none_func = cast(
                (Callable[[str, Tensor | None], T | SkippedCalc | None]), func
            )
        if return_set:
            return {
                result
                for k, v in self.items_flat(yield_none=True)
                if not isinstance(result := maybe_returns_none_func(k, v), SkippedCalc)
            }

        return {
            k: result
            for k, v in self.items_flat(yield_none=True)
            if not isinstance(result := maybe_returns_none_func(k, v), SkippedCalc)
        }

    @overload
    def items_flat(self) -> Iterator[tuple[str, Tensor]]: ...
    @overload
    def items_flat(
        self, yield_none: Literal[False]
    ) -> Iterator[tuple[str, Tensor]]: ...
    @overload
    def items_flat(
        self, yield_none: Literal[True]
    ) -> Iterator[tuple[str, Tensor | None]]: ...
    @overload
    def items_flat(
        self, yield_none: bool = False
    ) -> Iterator[tuple[str, Tensor]] | Iterator[tuple[str, Tensor | None]]: ...

    def items_flat(
        self, yield_none: bool = False
    ) -> Iterator[tuple[str, Tensor]] | Iterator[tuple[str, Tensor | None]]:
        for k, v in self.items():
            if isinstance(v, Tensor):
                yield k, v
            elif isinstance(v, DictBatch):
                for dk, dv in v.items_flat():
                    yield f"{k}.{dk}", dv
            elif v is not None:
                raise TypeError(f"Unexpected value type in DictBatch: {type(v)}")
            elif yield_none:
                yield k, v

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
            include = set(self.keys()) - (exclude or set())

        split = self.set_split(include)
        split.a = split.a[mask]
        return split.recombine()

    def split(
        self,
        split_size: int | list[int],
        dim: int = 0,
    ) -> list[Self]:
        assert dim == 0
        if isinstance(split_size, int):
            sizes = [split_size] * (self.batch_size // split_size)
            rem = self.batch_size % split_size
            if rem:
                sizes.append(rem)
        else:
            sizes = list(split_size)
            if sum(sizes) != self.batch_size:
                raise ValueError(
                    f"Split sizes must sum to batch size ({self.batch_size}), "
                    f"got {sizes}"
                )

        out: list[Self] = []
        start = 0
        for size in sizes:
            out.append(self[start : start + size])
            start += size
        return out

    def einops_rearrange(self, pattern: str, **kwargs):
        import einops

        def rearrange(t: Tensor) -> Tensor:
            return einops.rearrange(t, pattern, **kwargs)

        return self.apply_func(rearrange)

    def get_unanimous_result[T](self, func: Callable[[Tensor], T]) -> T:
        results = {func(t) for t in self.iter_tensors()}
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
        shapes: dict[str, tuple[int, ...]] = {}

        def add(prefix: str, v: Any):
            if v is None:
                return
            if isinstance(v, Tensor):
                shapes[prefix] = tuple(v.shape)
                return
            if isinstance(v, DictBatch):
                for k, shape in v.shapes.items():
                    shapes[f"{prefix}.{k}"] = shape
                return
            raise TypeError(f"Unexpected value type in DictBatch: {type(v)}")

        for k, v in self.items():
            add(k, v)

        return shapes

    @property
    def shape(self) -> "DictBatchShape":
        return DictBatchShape(batch_size=self.batch_size, shapes=self.shapes)

    def updated_with(self, **kwargs: Any) -> Self:
        return self.construct_with_other_data(
            {**self, **kwargs}, self._get_other_dict()
        )

    @classmethod
    def class_nested_dictbatch_fields(cls) -> dict[str, type["DictBatch"]]:
        db_fields = {
            k for k, v in cls.TENSOR_DATA_FIELDS.items() if _hint_allows_dictbatch(v)
        }
        tensor_fields = {
            k for k, v in cls.TENSOR_DATA_FIELDS.items() if _hint_allows_tensor(v)
        }
        assert len(db_fields & tensor_fields) == 0
        return {
            k: _get_dictbatch_constructor_type(v)
            for k, v in cls.TENSOR_DATA_FIELDS.items()
            if k in db_fields
        }

    @classmethod
    def cast_convert(
        cls,
        batch: "DictBatch | Mapping[str, Tensor | DictBatch | None]",
        **kwargs,
    ) -> Self:
        if isinstance(batch, cls) and type(batch) is cls and not kwargs:
            return batch

        def data_pred(k: str, v: Any) -> bool:
            return (
                isinstance(v, Tensor | DictBatch)
                or k in cls.TENSOR_DATA_FIELDS
                or k.split(".")[0] in cls.TENSOR_DATA_FIELDS
            )

        nested_fields = cls.class_nested_dictbatch_fields()

        def nested_data_pred(k: str, v: Any) -> bool:
            return k.split(".")[0] in nested_fields

        def other_pred(k: str, v: Any) -> bool:
            return k.split(".")[0] in cls.OTHER_DATA_FIELDS

        should_be_empty = {
            k: v for k, v in kwargs.items() if data_pred(k, v) and other_pred(k, v)
        }
        assert should_be_empty == {}

        extra_other_data = {k: v for k, v in kwargs.items() if other_pred(k, v)}
        flat_other = {
            k: v for k, v in extra_other_data.items() if not nested_data_pred(k, v)
        }
        flattened_nested_other = {
            k: v for k, v in extra_other_data.items() if nested_data_pred(k, v)
        }

        extra_data = {k: v for k, v in kwargs.items() if data_pred(k, v)}

        batch_data = {**_flatten(batch), **extra_data}
        flat_data = {k: v for k, v in batch_data.items() if not nested_data_pred(k, v)}
        flattened_nested_data = {
            k: v for k, v in batch_data.items() if nested_data_pred(k, v)
        }
        dictbatch_fields = cls.class_nested_dictbatch_fields()
        nested_data: dict[str, dict[str, TensorFieldTypes] | DictBatch] = (
            _unflatten_once_strict(flattened_nested_data)
        )
        nested_other: dict[str, dict[str, Any]] = _unflatten_once_strict(
            flattened_nested_other
        )
        for k in nested_data.keys():
            if k not in nested_other:
                nested_other[k] = {}
        nested_cast_data: dict[str, DictBatch] = {
            k: (
                dictbatch_fields[k] if k in dictbatch_fields else DictBatch
            ).cast_convert(v, **nested_other[k])
            for k, v in nested_data.items()
        }

        assert (
            len(
                set(kwargs.keys())
                - set(cls.TENSOR_DATA_FIELDS.keys())
                - set(cls.OTHER_DATA_FIELDS.keys())
            )
            == 0
        ), f"""Unexpected keys in kwargs: 
            {
            set(kwargs.keys())
            - set(cls.TENSOR_DATA_FIELDS.keys())
            - set(cls.OTHER_DATA_FIELDS.keys())
        }"""
        if isinstance(batch, DictBatch):
            other_data = {**batch._get_other_dict(), **flat_other}
        else:
            other_data = flat_other
        assert not (nested_cast_data.keys() & flat_data.keys()), (
            "Nested and flat data keys must be disjoint:"
            f"{nested_cast_data.keys() & flat_data.keys()} overlapped"
        )
        return cls.construct_with_other_data(
            {**nested_cast_data, **flat_data}, other_data
        )

    @overload
    def to_flat_dict(
        self, include_none: Literal[False] = False
    ) -> dict[str, Tensor]: ...
    @overload
    def to_flat_dict(self, include_none: Literal[True]) -> dict[str, Tensor | None]: ...
    @overload
    def to_flat_dict(
        self, include_none: bool = False
    ) -> dict[str, Tensor] | dict[str, Tensor | None]: ...
    def to_flat_dict(
        self, include_none: bool = False
    ) -> dict[str, Tensor] | dict[str, Tensor | None]:
        return dict(self.items_flat(yield_none=include_none))

    def save_as_safetensors(self, path: Path):
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)

        other_data = self._get_other_dict()

        if not all(isinstance(v, str) for v in other_data.values()):
            raise ValueError(
                "Cannot save DictBatch with non-string other data as safetensors"
            )

        save_file(self.to_flat_dict(), path, metadata=other_data)

    @classmethod
    def load_from_safetensors(cls, path: Path) -> Self:
        data, metadata = load_file_with_metadata(path)
        metadata = metadata or {}

        return cls.cast_convert(
            data,
            **metadata,
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
    # TENSOR_DATA_FIELDS = {
    #     "input_ids": Tensor,
    #     "attention_mask": Tensor,
    #     "labels": Tensor,
    # }
    # OTHER_DATA_FIELDS = {"prompt": str, "doc_id": int}

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
    # # should error:
    # batch = WithOptionalTensor(
    #     a=torch.ones(4, 128, dtype=torch.long),
    #     # b=None,
    # )

    # # should error:
    # batch = WithOptionalTensor(
    #     a=None,
    #     b=None,
    # )

    @DictBatch.auto_other_fields
    class WithOptionalTensorNest(DictBatch):
        n: WithOptionalTensor
        a: Tensor = dictbatch_field(factory=ones_maker)

    nested = WithOptionalTensorNest(
        n=WithOptionalTensor(
            a=torch.ones(4, 128, dtype=torch.long),
            b=torch.ones(4, 128, dtype=torch.long),
            c=torch.ones(4, 128, dtype=torch.long),
        ),
        a=torch.ones(4, 128, dtype=torch.long),
    )
    print(nested)
    print(nested.batch_size)

    # =========================================================================
    # Example 2: Nested DictBatch support
    # =========================================================================
    print("\n" + "=" * 60)
    print("Nested DictBatch Example")
    print("=" * 60)

    # Define an inner batch type
    @DictBatch.auto_other_fields
    class InnerBatch(DictBatch):
        hidden: Tensor
        activations: Tensor | None = None

    # Define an outer batch type that contains the inner batch
    @DictBatch.auto_other_fields
    class OuterBatch(DictBatch):
        tokens: Tensor
        inner: InnerBatch
        optional_inner: InnerBatch | None = None
        label: str

    # Create nested batches
    inner1 = InnerBatch(
        hidden=torch.randn(4, 64),
        activations=torch.randn(4, 32),
    )
    print(f"Inner batch size: {inner1.batch_size}")

    outer = OuterBatch(
        tokens=torch.randint(0, 1000, (4, 16)),
        inner=inner1,
        label="test",
    )
    print(f"Outer batch size: {outer.batch_size}")
    print(f"Outer TENSOR_DATA_FIELDS: {outer.TENSOR_DATA_FIELDS}")

    # Access nested batch
    print(f"outer.inner.hidden.shape: {outer.inner.hidden.shape}")
    print(f"outer['inner'].hidden.shape: {outer['inner'].hidden.shape}")

    # Indexing propagates to nested batches
    indexed = outer[:2]
    print("\nAfter indexing [:2]:")
    print(f"  indexed.batch_size: {indexed.batch_size}")
    print(f"  indexed.tokens.shape: {indexed.tokens.shape}")
    print(f"  indexed.inner.hidden.shape: {indexed.inner.hidden.shape}")

    # .to() propagates to nested batches
    if torch.cuda.is_available():
        outer_cuda = outer.cuda()
        print("\nAfter .cuda():")
        print(f"  outer_cuda.tokens.device: {outer_cuda.tokens.device}")
        print(f"  outer_cuda.inner.hidden.device: {outer_cuda.inner.hidden.device}")

    # cat_list works with nested batches
    outer2 = OuterBatch(
        tokens=torch.randint(0, 1000, (4, 16)),
        inner=InnerBatch(
            hidden=torch.randn(4, 64),
            activations=torch.randn(4, 32),
        ),
        label="test",
    )
    a = outer.to_flat_dict(include_none=True)
    b = _flatten(outer)
    a.keys() ^ b.keys()
    concatenated = OuterBatch.cat_list([outer, outer2], dim=0)
    print("\nAfter cat_list:")
    print(f"  concatenated.batch_size: {concatenated.batch_size}")
    print(f"  concatenated.tokens.shape: {concatenated.tokens.shape}")
    print(f"  concatenated.inner.hidden.shape: {concatenated.inner.hidden.shape}")

    # clone() propagates to nested batches
    cloned = outer.clone()
    print("\nAfter clone():")
    print(f"  cloned.inner is outer.inner: {cloned.inner is outer.inner}")
    print(
        f"  cloned.inner.hidden is outer.inner.hidden: {cloned.inner.hidden is outer.inner.hidden}"
    )
    cloned.items()

    print("\nNested DictBatch tests passed!")
