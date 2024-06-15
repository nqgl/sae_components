from saeco.core.cache import Cache


import torch


from abc import ABC, abstractmethod


from typing import Callable, List, Any

c2t = Callable[..., torch.Tensor | List[torch.Tensor]]


class Module(torch.nn.Module, ABC):
    __call__: c2t

    @abstractmethod
    def forward(self, x, *, cache: Cache, **kwargs):
        raise NotImplementedError
