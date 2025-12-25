from collections.abc import Callable
from typing import ClassVar

import torch

from saeco.data.dict_batch.dict_batch import DictBatch


def expand_to(a: torch.Tensor, b: torch.Tensor):
    assert a.ndim <= b.ndim
    for _ in range(b.ndim - a.ndim):
        a = a.unsqueeze(-1)
    return a


class DictBatchWithSeq(DictBatch):
    non_seq_tensors: ClassVar[tuple[str, ...]] = ()

    def _get_non_seq_tensors(
        self, non_seq_tensors: list[str] | tuple[str, ...] | None = None
    ) -> tuple[str, ...]:
        if non_seq_tensors is not None:
            return tuple(non_seq_tensors)
        if self.non_seq_tensors is not None:
            return self.non_seq_tensors
        raise ValueError("No non-seq tensors provided")

    def gather_along_seq(
        self,
        perm: torch.Tensor,
        non_seq_tensors: tuple[str, ...] | list[str] | None = None,
    ):
        def gather(t: torch.Tensor) -> torch.Tensor:
            return t.gather(dim=1, index=expand_to(perm, t).expand_as(t))

        return self.apply_to_seq_tensors(gather, non_seq_tensors=non_seq_tensors)

    def index_along_seq(
        self,
        indexer: torch.Tensor | slice | int,
        non_seq_tensors: tuple[str, ...] | list[str] | None = None,
    ):
        def index(t: torch.Tensor) -> torch.Tensor:
            return t[:, indexer]

        return self.apply_to_seq_tensors(index, non_seq_tensors=non_seq_tensors)

    def apply_to_seq_tensors(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        *,
        non_seq_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        non_seq_tensors: tuple[str, ...] | list[str] | None = None,
    ):
        non_seq_tensors = self._get_non_seq_tensors(non_seq_tensors)
        split = self.set_split(non_seq_tensors)
        split.a = split.a.apply_func(non_seq_func)
        split.b = split.b.apply_func(func)

        return split.recombine()
