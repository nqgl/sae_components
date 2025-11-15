from typing import Callable, ContextManager, Protocol

import torch

from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.trainer.trainable import Trainable


class ReconstructionEvaluatorFunctionProtocol(Protocol):
    def __call__(
        self,
        llm,
        sae: Trainable,
        tokens,
        cfg: ActsDataConfig,
        cast_fn: Callable[[], ContextManager] = ...,
        num_batches=10,
        batch_size=1,
    ) -> dict[str, float]: ...
