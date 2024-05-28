from sae_components.core.collections.parallel import (
    AddParallel,
    MulParallel,
    Parallel,
)


import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class Mul(MulParallel):
    """
    Multiplies the input by mul.
    """

    multiplier: nn.Parameter

    #
    def __init__(self, mul):
        assert not isinstance(mul, nn.Module)
        super().__init__(
            identity=Identity(),
            multiplier=mul,
        )


class Add(AddParallel):
    """
    Adds bias to the input.
    """

    bias: nn.Parameter

    def __init__(self, bias):
        # assert not isinstance(bias, nn.Module)
        super().__init__(
            identity=Identity(),
            bias=bias,
        )


class Neg(Parallel):
    def __init__(self, item):
        super().__init__(
            negated_item=item,
            _support_modules=True,
            _support_parameters=True,
        )
        self.reduce(lambda x: -x)


class Sub(Add):
    """
    Subtracts bias from the input.
    Sub(bias) is equivalent to Add(Neg(bias)).
    """

    def __init__(self, bias):
        assert not isinstance(bias, nn.Module)
        super().__init__(
            bias=Neg(bias),
        )


class MatMul(Parallel):
    right: nn.Parameter

    def __init__(self, weight, *, weight_module=None):
        assert isinstance(weight, nn.Parameter)
        super().__init__(
            left=Identity(),
            right=weight,
            _support_parameters=True,
        )
        self.reduce(lambda input, weight: input @ weight)


from typing import Protocol, TypeVar
