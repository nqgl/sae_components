from saeco.core.cache import Cache
import torch


class SAECache(Cache):
    forward_reuse_dict: dict = ...
    sparsity_penalty: torch.NumberType = ...
