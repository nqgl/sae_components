from typing import Any

import torch

from saeco.core.cache import Cache


class SAECache(Cache):
    forward_reuse_dict: dict[tuple[int, int], Any] = ...
    sparsity_penalty: torch.Tensor = ...
