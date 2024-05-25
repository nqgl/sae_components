import torch.nn as nn
from sae_components.components.ops.fnlambda import Lambda
import sae_components.core as cl
import sae_components.core.module


# class Lambda(cl.Module):
#     def __init__(self, func):
#         super().__init__()
#         self.func = func

#     def forward(self, x, *, cache: cl.Cache, **kwargs):
#         return self.func(x)


class Detached(Lambda):
    def __init__(self, module):
        super().__init__(lambda x: x.detach(), module=module)


class Thresh(Lambda):
    def __init__(self, module):
        super().__init__(lambda x: (x > 0).float(), module=module)


def main():
    import torch

    thresh = Thresh(nn.Linear(10, 10))
    print(thresh(torch.randn(10), cache=cl.Cache()))


if __name__ == "__main__":
    main()
