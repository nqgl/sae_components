from saeco.core.cache import Cache


import torch


from abc import ABC, abstractmethod, ABCMeta


from typing import Callable, List, Any

c2t = Callable[..., torch.Tensor | List[torch.Tensor]]


class Module(torch.nn.Module, ABC):
    __call__: c2t

    @abstractmethod
    def forward(self, x, *, cache: Cache, **kwargs):
        raise NotImplementedError

    @classmethod
    def __instancecheck__(cls: ABCMeta, instance: torch.Any) -> bool:
        from saeco.components.wrap import WrapsModule

        return super().__instancecheck__(instance) or (
            isinstance(instance, WrapsModule) and isinstance(instance.wrapped, cls)
        )
