from collections.abc import Iterator

import torch
from torch import Tensor

from saeco.evaluation.filtered import FilteredTensor
from saeco.evaluation.named_filter import NamedFilter
from saeco.evaluation.storage.chunk import Chunk


class MetadataBuilder:
    def __init__(self, chunks, dtype, device, shape):
        self.it = iter(chunks)
        self.chunks = chunks
        self._value = torch.zeros(*shape, dtype=dtype, device=device)
        self.done = False
        self.chunks_done = [False] * len(chunks)
        self.i = 0
        self.unique_labels = {}  # for strings only

    @property
    def value(self):
        self.finish()
        return self._value

    def finish(self):
        assert all(self.chunks_done)
        self.done = True

    def __iter__(self) -> Iterator[Chunk]:
        return self

    def __next__(self) -> Chunk:
        return next(self.it)

    def __lshift__(self, v):
        return self._recv(self.chunks[self.i], v)

    def _recv(self, chunk: Chunk, value: FilteredTensor | Tensor):
        assert isinstance(chunk, Chunk)
        assert isinstance(value, FilteredTensor | Tensor)
        assert not self.done
        assert not self.chunks_done[chunk.idx]
        if isinstance(value, Tensor):
            value = chunk._to_filtered(value)
        value.filter.writeat(target=self._value, value=value.value)
        self.chunks_done[chunk.idx] = True
        self.i += 1

    def takestrl(self, v):
        assert isinstance(v, list) and self._value.dtype == torch.long
        o = [self.getlabel(s) for s in v]
        t = torch.tensor(o, dtype=torch.long, device=self._value.device)
        self << t

    def getlabel(self, s: str):
        assert isinstance(s, str)
        if s in self.unique_labels:
            return self.unique_labels[s]
        self.unique_labels[s] = len(self.unique_labels)
        return self.unique_labels[s]

    class mbsetter:
        def __init__(self, mb, chunk):
            self.mb: MetadataBuilder = mb
            self.chunk: Chunk = chunk

        def __lshift__(self, v):
            return self.mb._recv(self.chunk, v)

    def __getitem__(self, chunk):
        return MetadataBuilder.mbsetter(self, chunk)

    def __setitem__(self, chunk, value):
        return self._recv(chunk, value)


class FilteredBuilder:
    def __init__(
        self,
        chunks: list[Chunk],
        dtype,
        device,
        shape: tuple[int],
        filter: NamedFilter,
    ):
        self.it = iter(chunks)
        self.chunks = chunks
        self.done = False
        self.chunks_done = [False] * len(chunks)
        self.i = 0
        self.unique_labels = {}  # for strings only
        self.filter = filter
        assert filter is None or shape[0] == filter.filter.shape[0]
        self._value = torch.zeros(*shape, dtype=dtype, device=device)

    @property
    def value(self) -> FilteredTensor:
        self.finish()
        return self._value

    def finish(self):
        assert all(self.chunks_done)
        self.done = True
        self._value = FilteredTensor.from_unmasked_value(self._value, self.filter)

    def __iter__(self) -> Iterator[Chunk]:
        return self

    def __next__(self) -> Chunk:
        return next(self.it)

    def __lshift__(self, v):
        return self._recv(self.chunks[self.i], v)

    def _recv(self, chunk: Chunk, value: FilteredTensor | Tensor):
        assert isinstance(chunk, Chunk)
        assert isinstance(value, FilteredTensor | Tensor)
        assert not self.done
        assert not self.chunks_done[chunk.idx]
        if isinstance(value, Tensor):
            value = chunk._to_filtered(value)
        value.filter.writeat(target=self._value, value=value.value)
        self.chunks_done[chunk.idx] = True
        self.i += 1

    def takestrl(self, v):
        assert isinstance(v, list) and self._value.dtype == torch.long
        o = [self.getlabel(s) for s in v]
        t = torch.tensor(o, dtype=torch.long, device=self._value.device)
        self << t

    def getlabel(self, s: str):
        assert isinstance(s, str)
        if s in self.unique_labels:
            return self.unique_labels[s]
        self.unique_labels[s] = len(self.unique_labels)
        return self.unique_labels[s]

    class mbsetter:
        def __init__(self, mb, chunk):
            self.mb: MetadataBuilder = mb
            self.chunk: Chunk = chunk

        def __lshift__(self, v):
            return self.mb._recv(self.chunk, v)

    def __getitem__(self, chunk):
        return MetadataBuilder.mbsetter(self, chunk)

    def __setitem__(self, chunk, value):
        return self._recv(chunk, value)
