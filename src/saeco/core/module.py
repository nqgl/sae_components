from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

from saeco.core.cache import Cache

c2t = Callable[..., torch.Tensor | list[torch.Tensor]]


class Module(torch.nn.Module, ABC):
    __call__: c2t

    @abstractmethod
    def forward(self, x, *, cache: Cache, **kwargs):
        raise NotImplementedError

    @classmethod
    def __instancecheck__(cls: ABCMeta, instance: Any) -> bool:
        from saeco.components.wrap import WrapsModule

        return super().__instancecheck__(instance) or (
            isinstance(instance, WrapsModule) and isinstance(instance.wrapped, cls)
        )

    @contextmanager
    def train_ctx(self):
        prev_state = self.training
        self.train()
        try:
            yield
        finally:
            self.train(prev_state)

    @contextmanager
    def eval_ctx(self):
        prev_state = self.training
        self.eval()
        try:
            yield
        finally:
            self.train(prev_state)
