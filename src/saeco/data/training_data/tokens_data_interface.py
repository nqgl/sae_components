from abc import ABC, abstractmethod

import torch
from saeco.data.dict_batch import DictBatch


class TokensDataInterface[DataT: torch.Tensor | DictBatch](ABC):
    @property
    @abstractmethod
    def seq_len(self) -> int: ...

    @property
    @abstractmethod
    def num_tokens(self) -> int: ...

    @abstractmethod
    def get_tokens(self, num_tokens: int | None = None) -> DataT: ...
