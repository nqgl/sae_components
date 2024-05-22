import torch.nn as nn
import sae_components.core as cl
import sae_components.core.module


class Lambda(cl.Module):
    def __init__(self, module, func):
        super().__init__()
        self.module = module
        self.func = func

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        if isinstance(self.module, cl.Module):
            return self.func(self.module(x, cache=cache, **kwargs))
        return self.func(self.module(x))


# class Lambda(cl.Module):
#     def __init__(self, func):
#         super().__init__()
#         self.func = func

#     def forward(self, x, *, cache: cl.Cache, **kwargs):
#         return self.func(x)


# class Detach(Lambda):
#     def __init__(self):
#         super().__init__(lambda x: x.detach())


class Thresh(Lambda):
    def __init__(self, module):
        super().__init__(module, lambda x: (x > 0).float())


def main():
    import torch

    thresh = Thresh(nn.Linear(10, 10))
    print(thresh(torch.randn(10), cache=cl.Cache()))


if __name__ == "__main__":
    main()