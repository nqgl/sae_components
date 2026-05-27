from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager

import torch

from saeco.core.cache import Cache

c2t = Callable[..., torch.Tensor | list[torch.Tensor]]


class Module(torch.nn.Module, ABC):
    __call__: c2t

    @abstractmethod
    def forward(self, x, *, cache: Cache, **kwargs):
        raise NotImplementedError

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
