from abc import ABCMeta

import torch
import torch.nn as nn
from typing import Any

import saeco.core as cl


class WrapsModule(cl.Module):
    wrapped: nn.Module

    def __init__(self, module):
        super().__init__()
        try:
            object.__getattribute__(self, "module")
            raise ValueError("module attribute already exists on self")
        except AttributeError:
            pass
        object.__setattr__(self, "module", module)
        # self._parameters = module._parameters.copy()
        # self._buffers = module._buffers.copy()
        # self._modules = module._modules.copy()
        self.wrapped = module
        # self.__class__ = type(
        #     f"{self.__class__.__name__}[{module.__class__.__name__}]",
        #     (self.__class__, module.__class__),
        #     {},
        # )
        # self.register_buffer("module", module)

    def _get_name(self):
        return f"{self.__class__.__name__}[{self.wrapped._get_name()}]"

    def forward(self, *args, **kwargs):  # TODO do this differently less jank
        if isinstance(self.wrapped, cl.Module):
            return self.wrapped(*args, **kwargs)
        kwargs.pop("cache", None)
        return self.wrapped(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("wrapped"), name)

    @classmethod
    def __instancecheck__(cls: ABCMeta, instance: Any) -> bool:
        return super().__instancecheck__(instance) or (
            isinstance(instance, WrapsModule) and isinstance(instance.wrapped, cls)
        )


def main():
    import saeco.core as cl
    from saeco.core.basic_ops import Add, Mul, Neg, Sub

    seq = cl.Seq(
        Add(nn.Parameter(torch.tensor(1.0))),
        Mul(nn.Parameter(torch.tensor(2.0))),
        Sub(nn.Parameter(torch.tensor(3.0))),
    )

    w = WrapsModule(seq)
    w2 = WrapsModule(
        cl.Seq(
            Add(nn.Parameter(torch.tensor(1.0))),
            Mul(nn.Parameter(torch.tensor(2.0))),
            Sub(nn.Parameter(torch.tensor(3.0))),
        )
    )
    cache = cl.Cache()
    print(w)
    print(w(2, cache=cache))

    class A(nn.Module):
        def __init__(self, module=None):
            super().__init__()
            if module is None:
                self.p0 = nn.Parameter(torch.tensor(1.0))
            else:
                self.p = nn.Parameter(torch.tensor(1.0))
                self.submodule = module

        def forward(self, x):
            return x

        def post_backward_hook(self):
            print("A")

    class BMix(WrapMix):
        def post_backward_hook(self):
            print("BMix")

    class Bwm(WrapsModule):
        def post_backward_hook(self):
            print("Bwm")

    from saeco.trainer.call_training_hooks import do_post_backward

    a = cl.Seq(
        Add(nn.Parameter(torch.tensor(1.0))),
        Mul(nn.Parameter(torch.tensor(2.0))),
        Sub(nn.Parameter(torch.tensor(3.0))),
    )
    # a = A()
    model1 = BMix(a)
    model2 = Bwm(a)
    print(model1)
    print("a")
    # model1.apply(post_backward)
    print(model2)
    model2.apply(do_post_backward)
    www = Bwm(Bwm(Bwm(A(a))))
    # www = BMix(BMix(BMix(A(a))))

    print(www)
    www.apply(do_post_backward)
    print(www.wrapped)
    print(www.wrapped.wrapped.wrapped)
    print([p for p in www.named_parameters()])
    # print(model1)


if __name__ == "__main__":
    main()
