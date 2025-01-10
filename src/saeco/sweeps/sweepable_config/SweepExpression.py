from saeco.sweeps.sweepable_config.Swept import Swept


from types import NoneType
from typing import Any, TypeVar, get_args, get_origin, Generic

T = TypeVar("T")
LITERALS = [int, float, str, bool]


class SweepExpression(Swept[T], Generic[T]):
    values: list = []

    @classmethod
    def convert_other(cls, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import (
            ExpressionOpEnum,
            Op,
            Val,
        )

        if isinstance(other, SweepExpression):
            return other
        if any([isinstance(other, l) for l in LITERALS]):
            return Val[type(other)](value=other)
        # if isinstance(other, SweepVar):
        #     return Var(name=other.name, sweep_var=other)
        raise ValueError(f"Cannot convert {other} to SweepExpressionNode")

    @classmethod
    def common_type(cls, objs: list["SweepExpression"]):
        t = None
        for o in objs:
            if o.generic_type is not None:
                if t is None:
                    t = o.generic_type
                elif o.generic_type is float and t is int:
                    t = float
                elif t is float and o.generic_type is int:
                    t = float
                elif issubclass(t, o.generic_type):
                    t = o.generic_type
                elif issubclass(o.generic_type, t):
                    pass
                else:
                    raise ValueError(
                        f"Cannot find common type for {t} and {o.generic_type}"
                    )
        return t

    def __mul__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import (
            ExpressionOpEnum,
            Op,
            Val,
        )

        other = self.convert_other(other)
        type(self)
        type(other)

        t = self.common_type([self, other])
        return Op[t](op=ExpressionOpEnum.MUL, children=[self, other])

    def __add__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import (
            ExpressionOpEnum,
            Op,
            Val,
        )

        other = self.convert_other(other)
        t = self.common_type([self, other])
        return Op[t](op=ExpressionOpEnum.ADD, children=[self, other])

    def __sub__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import (
            ExpressionOpEnum,
            Op,
            Val,
        )

        other = self.convert_other(other)
        t = self.common_type([self, other])
        return Op[t](op=ExpressionOpEnum.SUB, children=[self, other])

    def __truediv__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import (
            ExpressionOpEnum,
            Op,
            Val,
        )

        other = self.convert_other(other)
        t = self.common_type([self, other])
        return Op[float](op=ExpressionOpEnum.FLOATDIV, children=[self, other])

    def __floordiv__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import (
            ExpressionOpEnum,
            Op,
            Val,
        )

        other = self.convert_other(other)
        t = self.common_type([self, other])
        return Op[int](op=ExpressionOpEnum.INTDIV, children=[self, other])

    def __pow__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import (
            ExpressionOpEnum,
            Op,
            Val,
        )

        other = self.convert_other(other)
        t = self.common_type([self, other])
        return Op[t](op=ExpressionOpEnum.POW, children=[self, other])

    def __mod__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import (
            ExpressionOpEnum,
            Op,
            Val,
        )

        other = self.convert_other(other)
        t = self.common_type([self, other])
        return Op[t](op=ExpressionOpEnum.MOD, children=[self, other])

    def __getitem__(self, other):
        from saeco.sweeps.sweepable_config.sweep_expressions import (
            ExpressionOpEnum,
            Op,
            Val,
        )

        get_origin(type(self))
        get_args(type(self))
        other = self.convert_other(other)

        # if (
        #     "args" in self.__pydantic_generic_metadata__
        #     and self.__pydantic_generic_metadata__["args"]
        # ):
        dtype = self.generic_type
        if get_origin(dtype):
            if get_origin(dtype) is dict:
                t = get_args(dtype)[1]
            elif get_origin(dtype) in [list, tuple]:
                t = get_args(dtype)[0]
            else:
                raise ValueError(f"Cannot index {dtype}")
        else:
            t = get_args(dtype)[1]
        # else:
        #     t = get_args(self.generic_type)[1]
        return Op[t](op=ExpressionOpEnum.INDEX, children=[self, other])

    def evaluate(self, vars_dict: dict[str, Any]): ...

    def get_sweepvars(self):
        raise NotImplementedError(f"get_sweepvars not implemented for {type(self)}")
