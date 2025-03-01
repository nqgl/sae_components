from enum import Enum

from pydantic import BaseModel
from saeco.sweeps.sweepable_config.sweepable_config import SweepVar, Swept, T


from typing import Any, Callable


# class SweepExpression(Swept[T]):
#     expr: "SweepExpressionNode"

#     def __init__(self, *args, expr: Callable[[], T], **kwargs):
#         super().__init__(values=[], expr=expr, args=args, kwargs=kwargs)

#     def get_sweepvars(self) -> set["SweepVar"]:
#         params = set()
#         for p in self.args + tuple(self.kwargs.values()):
#             if isinstance(p, SweepVar):
#                 params.add(p)
#             elif isinstance(p, SweepExpression):
#                 params |= p.get_sweepvars()
#         return params

#     def evaluate(self, *args, **kwargs):
#         print(f"evaluating SweepExpression {self.expr}(*{args}, **{kwargs})")
#         return self.expr(*args, **kwargs)

#     def instantiate(self):
#         return self.evaluate(*self.get_args(), **self.get_kwargs())

#     def instantiated(self, swept_vars):
#         return self.evaluate(*self.get_args(swept_vars), **self.get_kwargs(swept_vars))

#     def get_args(self, swept_vars=None):
#         if swept_vars is None:
#             return [
#                 a.instantiated_value if isinstance(a, SweepVar) else a
#                 for a in self.args
#             ]
#         return [
#             swept_vars[arg.name] if isinstance(arg, SweepVar) else arg
#             for arg in self.args
#         ]

#     def get_kwargs(self, swept_vars=None):
#         if swept_vars is None:
#             return {
#                 k: v.instantiated_value if isinstance(v, SweepVar) else v
#                 for k, v in self.kwargs.items()
#             }
#         return {
#             k: swept_vars[kwarg.name] if isinstance(kwarg, SweepVar) else kwarg
#             for k, kwarg in self.kwargs.items()
#         }


# class SweepExpression(Swept[T]):
#     expr: "SweepExpressionNode"

#     def __init__(self, *args, expr: Callable[[], T], **kwargs):
#         super().__init__(values=[], expr=expr, args=args, kwargs=kwargs)

#     def get_sweepvars(self) -> set["SweepVar"]:
#         params = set()
#         for p in self.args + tuple(self.kwargs.values()):
#             if isinstance(p, SweepVar):
#                 params.add(p)
#             elif isinstance(p, SweepExpression):
#                 params |= p.get_sweepvars()
#         return params

#     def evaluate(self, *args, **kwargs):
#         print(f"evaluating SweepExpression {self.expr}(*{args}, **{kwargs})")
#         return self.expr(*args, **kwargs)

#     def instantiate(self):
#         return self.evaluate(*self.get_args(), **self.get_kwargs())

#     def instantiated(self, swept_vars):
#         return self.evaluate(*self.get_args(swept_vars), **self.get_kwargs(swept_vars))

#     def get_args(self, swept_vars=None):
#         if swept_vars is None:
#             return [
#                 a.instantiated_value if isinstance(a, SweepVar) else a
#                 for a in self.args
#             ]
#         return [
#             swept_vars[arg.name] if isinstance(arg, SweepVar) else arg
#             for arg in self.args
#         ]

#     def get_kwargs(self, swept_vars=None):
#         if swept_vars is None:
#             return {
#                 k: v.instantiated_value if isinstance(v, SweepVar) else v
#                 for k, v in self.kwargs.items()
#             }
#         return {
#             k: swept_vars[kwarg.name] if isinstance(kwarg, SweepVar) else kwarg
#             for k, kwarg in self.kwargs.items()
#         }


class SweepExpressionNode(BaseModel):
    @classmethod
    def convert_other(cls, other):
        if isinstance(other, SweepExpressionNode):
            return other
        if any([isinstance(other, l) for l in LITERALS]):
            return Val(value=other)
        if isinstance(other, SweepVar):
            return Var(name=other.name)
        raise ValueError(f"Cannot convert {other} to SweepExpressionNode")

    def __mul__(self, other):
        other = self.convert_other(other)
        return Op(op=ExpressionOpEnum.MUL, left=self, right=other)

    def __add__(self, other):
        other = self.convert_other(other)
        return Op(op=ExpressionOpEnum.ADD, left=self, right=other)

    def __sub__(self, other):
        other = self.convert_other(other)
        return Op(op=ExpressionOpEnum.SUB, left=self, right=other)

    def __truediv__(self, other):
        other = self.convert_other(other)
        return Op(op=ExpressionOpEnum.FLOATDIV, left=self, right=other)

    def __floordiv__(self, other):
        other = self.convert_other(other)
        return Op(op=ExpressionOpEnum.INTDIV, left=self, right=other)

    def __pow__(self, other):
        other = self.convert_other(other)
        return Op(op=ExpressionOpEnum.POW, left=self, right=other)

    def __mod__(self, other):
        other = self.convert_other(other)
        return Op(op=ExpressionOpEnum.MOD, left=self, right=other)

    def evaluate(self, vars_dict: dict[str, Any]): ...

    def get_sweepvars(self): ...


class Op(SweepExpressionNode):
    op: ExpressionOpEnum
    children: list[SweepVar | "SweepExpressionNode"]

    def evaluate(self, vars_dict: dict[str, Any]):
        return self.op.evaluate(*[child.evaluate(vars_dict) for child in self.children])

    def get_sweepvars(self):
        s = set()
        for child in self.children:
            s |= child.get_sweepvars()
        return s


class Var(SweepExpressionNode):
    name: str

    def evaluate(self, vars_dict: dict[str, Any]):
        return vars_dict[self.name]

    def get_sweepvars(self):
        # shou;d this ret the name or the sweepvar object do we need
        # to assess the paradign of the sweepvar class
        ...


class Val(SweepExpressionNode):
    value: str | int | float | bool

    def evaluate(self, vars_dict: dict[str, Any]):
        return self.value


class SweepExpression(Swept[T]):
    expr: "SweepExpressionNode"

    def __init__(self, *args, expr: Callable[[], T], **kwargs):
        super().__init__(values=[], expr=expr, args=args, kwargs=kwargs)

    def get_sweepvars(self) -> set["SweepVar"]:
        params = set()
        for p in self.args + tuple(self.kwargs.values()):
            if isinstance(p, SweepVar):
                params.add(p)
            elif isinstance(p, SweepExpression):
                params |= p.get_sweepvars()
        return params

    def evaluate(self, *args, **kwargs):
        print(f"evaluating SweepExpression {self.expr}(*{args}, **{kwargs})")
        return self.expr(*args, **kwargs)

    def instantiate(self):
        return self.evaluate(*self.get_args(), **self.get_kwargs())

    def instantiated(self, swept_vars):
        return self.evaluate(*self.get_args(swept_vars), **self.get_kwargs(swept_vars))

    def get_args(self, swept_vars=None):
        if swept_vars is None:
            return [
                a.instantiated_value if isinstance(a, SweepVar) else a
                for a in self.args
            ]
        return [
            swept_vars[arg.name] if isinstance(arg, SweepVar) else arg
            for arg in self.args
        ]

    def get_kwargs(self, swept_vars=None):
        if swept_vars is None:
            return {
                k: v.instantiated_value if isinstance(v, SweepVar) else v
                for k, v in self.kwargs.items()
            }
        return {
            k: swept_vars[kwarg.name] if isinstance(kwarg, SweepVar) else kwarg
            for k, kwarg in self.kwargs.items()
        }


class ExpressionOpEnum(str, Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    FLOATDIV = "/"
    INTDIV = "//"
    POW = "**"
    MOD = "%"

    def evaluate(self, *args):
        ### non-binary ops
        if self == ExpressionOpEnum.ADD:
            z = args[0]
            for arg in args[1:]:
                z += arg
            return z
        elif self == ExpressionOpEnum.MUL:
            n = 1
            for arg in args:
                n *= arg
            return n

        ### binary ops
        if len(args) != 2:
            if self in (
                ExpressionOpEnum.SUB,
                ExpressionOpEnum.FLOATDIV,
                ExpressionOpEnum.INTDIV,
                ExpressionOpEnum.POW,
                ExpressionOpEnum.MOD,
            ):
                raise ValueError(
                    f"Binary operator {self} requires two arguments, got {len(args)}"
                )
        if self == ExpressionOpEnum.SUB:
            return args[0] - args[1]
        elif self == ExpressionOpEnum.FLOATDIV:
            return args[0] / args[1]
        elif self == ExpressionOpEnum.INTDIV:
            return args[0] // args[1]
        elif self == ExpressionOpEnum.POW:
            return args[0] ** args[1]
        elif self == ExpressionOpEnum.MOD:
            return args[0] % args[1]


ASSOCIATIVE_OPS = {
    ExpressionOpEnum.ADD,
    ExpressionOpEnum.MUL,
}
LITERALS = [int, float, str, bool]
