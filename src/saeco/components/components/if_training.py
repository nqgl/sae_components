import saeco.core as cl


class IfTraining(cl.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        if self.training:
            return cache(self).module(x, **kwargs)
        return x
