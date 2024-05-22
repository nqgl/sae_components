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

    def __init__(self, mul):
        assert not isinstance(mul, nn.Module)
        super().__init__(
            identity=Identity(),
            mulparam=mul,
        )


class Add(AddParallel):
    """
    Adds bias to the input.
    """

    def __init__(self, bias):
        # assert not isinstance(bias, nn.Module)
        super().__init__(
            identity=Identity(),
            addparam=bias,
        )


class Neg(Parallel):
    def __init__(self, item):
        super().__init__(
            negated_item=item,
            support_modules=True,
            support_parameters=True,
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
    def __init__(self, weight):
        super().__init__(
            input=Identity(),
            weight=weight,
            support_parameters=True,
        )
        self.reduce(lambda x, y: x @ y)
