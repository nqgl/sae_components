from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor

from saeco_research.evaluation.filtered import FilteredTensor
from saeco_research.evaluation.named_filter import NamedFilter
from saeco_research.evaluation.storage.chunk import Chunk


class MetadataBuilder:
    """
    Build a full-dataset metadata tensor incrementally from chunks.

    Usage pattern:
        mb = eval.metadata_builder(dtype=torch.long, device="cpu")
        for chunk in mb:
            mb << chunk.tokens.value["my_metadata"]
        metadata = mb.value
    """

    def __init__(self, chunks: list[Chunk], dtype: torch.dtype, device, shape):
        self._chunks = chunks
        self._it = iter(chunks)
        self._value = torch.zeros(*shape, dtype=dtype, device=device)

        self._done = False
        self._chunks_done = [False] * len(chunks)
        self._cursor = 0

        # Used only when ingesting strings -> ids
        self.unique_labels: dict[str, int] = {}

    @property
    def value(self) -> Tensor:
        self.finish()
        return self._value

    def finish(self) -> None:
        if self._done:
            return
        if not all(self._chunks_done):
            missing = [i for i, ok in enumerate(self._chunks_done) if not ok]
            raise RuntimeError(
                f"MetadataBuilder not finished; missing chunks: {missing}"
            )
        self._done = True

    def __iter__(self) -> Iterator[Chunk]:
        return self

    def __next__(self) -> Chunk:
        return next(self._it)

    def __lshift__(self, v: FilteredTensor | Tensor) -> None:
        return self._recv(self._chunks[self._cursor], v)

    def _recv(self, chunk: Chunk, value: FilteredTensor | Tensor) -> None:
        if self._done:
            raise RuntimeError("Cannot write into a finished MetadataBuilder")
        if self._chunks_done[chunk.idx]:
            raise RuntimeError(f"Chunk {chunk.idx} already written")

        if isinstance(value, Tensor):
            value = chunk._to_filtered(value)

        # FilteredTensor writes into the full target.
        value.filter.writeat(target=self._value, value=value.value)
        self._chunks_done[chunk.idx] = True
        self._cursor += 1

    def takestrl(self, v: list[str]) -> None:
        if self._value.dtype != torch.long:
            raise TypeError("takestrl requires a long dtype target")
        ids = torch.tensor(
            [self.getlabel(s) for s in v], dtype=torch.long, device=self._value.device
        )
        self << ids

    def getlabel(self, s: str) -> int:
        if s in self.unique_labels:
            return self.unique_labels[s]
        new_id = len(self.unique_labels)
        self.unique_labels[s] = new_id
        return new_id

    class _Setter:
        def __init__(self, mb: MetadataBuilder, chunk: Chunk):
            self._mb = mb
            self._chunk = chunk

        def __lshift__(self, v: FilteredTensor | Tensor) -> None:
            self._mb._recv(self._chunk, v)

    def __getitem__(self, chunk: Chunk) -> _Setter:
        return MetadataBuilder._Setter(self, chunk)

    def __setitem__(self, chunk: Chunk, value: FilteredTensor | Tensor) -> None:
        self._recv(chunk, value)


class FilteredBuilder:
    """
    Like MetadataBuilder, but returns a FilteredTensor at the end by applying `NamedFilter`.

    This is intended for building metadata on filtered evaluations.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        dtype: torch.dtype,
        device,
        shape: tuple[int, ...],
        filter: NamedFilter,
    ):
        if filter is None:
            raise ValueError("FilteredBuilder requires a NamedFilter (got None)")

        if shape[0] != filter.filter.shape[0]:
            raise ValueError("FilteredBuilder shape[0] must match filter length")

        self._chunks = chunks
        self._it = iter(chunks)
        self._done = False
        self._chunks_done = [False] * len(chunks)
        self._cursor = 0

        self._filter = filter
        self._value = torch.zeros(*shape, dtype=dtype, device=device)
        self.unique_labels: dict[str, int] = {}

    @property
    def value(self) -> FilteredTensor:
        self.finish()
        # Wrap final tensor with filter
        return FilteredTensor.from_unmasked_value(
            self._value, filter_obj=self._filter, presliced=False
        )

    def finish(self) -> None:
        if self._done:
            return
        if not all(self._chunks_done):
            missing = [i for i, ok in enumerate(self._chunks_done) if not ok]
            raise RuntimeError(
                f"FilteredBuilder not finished; missing chunks: {missing}"
            )
        self._done = True

    def __iter__(self) -> Iterator[Chunk]:
        return self

    def __next__(self) -> Chunk:
        return next(self._it)

    def __lshift__(self, v: FilteredTensor | Tensor) -> None:
        return self._recv(self._chunks[self._cursor], v)

    def _recv(self, chunk: Chunk, value: FilteredTensor | Tensor) -> None:
        if self._done:
            raise RuntimeError("Cannot write into a finished FilteredBuilder")
        if self._chunks_done[chunk.idx]:
            raise RuntimeError(f"Chunk {chunk.idx} already written")

        if isinstance(value, Tensor):
            value = chunk._to_filtered(value)

        value.filter.writeat(target=self._value, value=value.value)
        self._chunks_done[chunk.idx] = True
        self._cursor += 1

    def takestrl(self, v: list[str]) -> None:
        if self._value.dtype != torch.long:
            raise TypeError("takestrl requires a long dtype target")
        ids = torch.tensor(
            [self.getlabel(s) for s in v], dtype=torch.long, device=self._value.device
        )
        self << ids

    def getlabel(self, s: str) -> int:
        if s in self.unique_labels:
            return self.unique_labels[s]
        new_id = len(self.unique_labels)
        self.unique_labels[s] = new_id
        return new_id
