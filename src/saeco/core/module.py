from saeco.core.cache import Cache


import torch


from abc import ABC, abstractmethod


class Module(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x, *, cache: Cache, **kwargs):
        raise NotImplementedError
