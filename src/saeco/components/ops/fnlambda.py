import saeco.core as cl
import torch.nn as nn
from saeco.components.wrap import WrapsModule


class Lambda(cl.Module):
    def __init__(self, func, module=None):
        super().__init__()
        self.module = module
        self.func = func

    def _get_name(self):
        return super()._get_name() + f"[{self.func.__name__}]"

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        if self.module is None:
            return self.func(x)
        if isinstance(self.module, nn.Parameter):
            return self.func(self.module)
        if isinstance(self.module, cl.Module):
            return self.func(self.module(x, cache=cache, **kwargs))
        return self.func(self.module(x))


# class LambdaWrap(WrapsModule):
#     def __init__(self, func, module):
#         super().__init__(Lambda(func, module=module))
