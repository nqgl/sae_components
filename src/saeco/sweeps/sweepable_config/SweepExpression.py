from typing import TYPE_CHECKING, Any, get_args, get_origin

from saeco.sweeps.sweepable_config.expressions_utils import (
    common_type,
    convert_other,
    shared_type,
)
from saeco.sweeps.sweepable_config.Swept import Swept

if TYPE_CHECKING:
    pass

LITERALS = [int, float, str, bool]


class SweepExpression[T](Swept[T]):
    values: list = []

    def repr(self, level=0):
        return f"{self!r}"

    def __mul__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.MUL(self, other)

    def __rmul__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.MUL(other, self)

    def __add__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.ADD(self, other)

    def __radd__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.ADD(other, self)

    def __sub__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.SUB(self, other)

    def __rsub__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.SUB(other, self)

    def __truediv__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.FLOATDIV(self, other)

    def __rtruediv__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.FLOATDIV(other, self)

    def __floordiv__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.INTDIV(self, other)

    def __rfloordiv__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.INTDIV(other, self)

    def __pow__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.POW(self, other)

    def __rpow__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.POW(other, self)

    def __mod__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.MOD(self, other)

    def __rmod__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum

        return ExpressionOpEnum.MOD(other, self)

    def __getitem__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import ExpressionOpEnum, Op

        other = convert_other(other)

        dtype = self.generic_type
        if get_origin(dtype):
            if get_origin(dtype) is dict:
                t = get_args(dtype)[1]
            elif get_origin(dtype) in [list, tuple]:
                t = get_args(dtype)[0]
            else:
                raise ValueError(f"Cannot index {dtype}")
        else:
            print(dtype)
            try:
                t = get_args(dtype)[1]
            except:
                if dtype in [list, tuple]:
                    t = shared_type(self.values)
                elif dtype is dict:
                    t = common_type([v for v in self.values.values()])
                else:
                    raise ValueError(f"Cannot index {dtype}")
        # else:
        #     t = get_args(self.generic_type)[1]
        return Op[t](op=ExpressionOpEnum.INDEX, children=[self, other])

    def evaluate(self, vars_dict: dict[str, Any]) -> T: ...

    def get_sweepvars(self):
        raise NotImplementedError(f"get_sweepvars not implemented for {type(self)}")
