import contextlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

import torch

from saeco.data.config.data_cfg import DataConfig
from saeco.trainer.trainable import Trainable


class ReconstructionEvaluatorFunctionProtocol(Protocol):
    def __call__(
        self,
        llm,
        sae: Trainable,
        tokens: torch.Tensor,
        data_cfg: DataConfig,
        cast_fn: Callable[[], AbstractContextManager] = contextlib.nullcontext,
        num_batches: int = 10,
        batch_size: int = 1,
    ) -> dict[str, float]: ...
