from sae_components.components.sae_cache import SAECache
import torch.nn as nn
import torch


class ReuseForward(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, cache: SAECache, **kwargs):
        key = (self.module, args)
        if not cache.has.forward_reuse_dict:
            cache.forward_reuse_dict = {}
        elif key in cache.forward_reuse_dict:
            return cache.forward_reuse_dict[key]
        output = self.module(*args, cache=cache, **kwargs)
        if cache.has.forward_reuse_dict:
            cache.forward_reuse_dict[key] = output
        return output
