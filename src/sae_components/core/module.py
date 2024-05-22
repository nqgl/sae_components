from sae_components.core.cache import Cache


import torch


from abc import ABC, abstractmethod


class Module(torch.nn.Module, ABC):
    @abstractmethod
    # TODO: Recently changed *x -> x
    def forward(self, x, *, cache: Cache, **kwargs):
        raise NotImplementedError
