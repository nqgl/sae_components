from enum import Enum

from pydantic import BaseModel
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config.Swept import Swept


from typing import Any, Callable, TypeVar, Union

from types import NoneType

T = TypeVar("T")


class ExpressionOpEnum(str, Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    FLOATDIV = "/"
    INTDIV = "//"
    POW = "**"
    MOD = "%"
    INDEX = "[]"

    def evaluate(self, *args):
        ### non-binary ops
        print("evaluating", self, args)
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
        elif self == ExpressionOpEnum.INDEX:
            if isinstance(args[0], dict):
                if args[1] in args[0]:
                    assert isinstance(args[1], str) or str(args[1]) not in args[0]
                    return args[0][args[1]]
                return args[0][str(args[1])]
            elif isinstance(args[0], list) or isinstance(args[0], tuple):
                return args[0][int(args[1])]


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


class SweepVar(SweepExpression[T]):
    values: list[T]
    name: str
    instantiated_value: T | None = None

    def __hash__(self):
        return hash(self.name)

    def sweep_dump(self):
        as_swept = Swept[T](*self.values)
        return as_swept.model_dump()

    @property
    def generic_type(self):
        try:  # take explicitly designated type if one is given
            t = super().generic_type
        except:
            t = None
        if t is not None:
            return t
        assert len(self.values) > 0
        types = {type(v) for v in self.values}
        if len(types) == 1:
            return types.pop()
        types = list(types)
        t = types[0]
        for typ in types[1:]:
            t = t | typ
        return t

    def get_sweepvars(self):
        return set([self])

    def evaluate(self, vars_dict: dict[str, Any]):
        return vars_dict[self.name]


class Op(SweepExpression):
    op: ExpressionOpEnum
    children: list[Union["Op", "Val", SweepVar]]
    values: list = []

    def evaluate(self, vars_dict: dict[str, Any]):
        print(self)
        print(vars_dict)
        return self.op.evaluate(*[child.evaluate(vars_dict) for child in self.children])

    def get_sweepvars(self):
        s = set()
        for child in self.children:
            s |= child.get_sweepvars()
        return s


# class Var(SweepExpression):
#     # name: str
#     sweep_var: SweepVar | None = None

#     def evaluate(self, vars_dict: dict[str, Any]):
#         return vars_dict[self.sweep_var.name]

#     def get_sweepvars(self):
#         # shou;d this ret the name or the sweepvar object do we need
#         # to assess the paradign of the sweepvar class
#         ...


class Val(SweepExpression):
    value: str | int | float | bool | list | tuple | dict

    def evaluate(self, vars_dict: dict[str, Any]):
        return self.value

    def get_sweepvars(self):
        return set()

    @property
    def generic_type(self):
        if not (
            isinstance(self.value, dict)
            or isinstance(self.value, list)
            or isinstance(self.value, tuple)
        ):
            return type(self.value)
        if isinstance(self.value, dict):
            kt = type(next(iter(self.value.keys())))
            vt = type(next(iter(self.value.values())))
            assert all(
                [isinstance(k, kt) and isinstance(v, vt) for k, v in self.value.items()]
            )
            return dict[kt, vt]
        elif isinstance(self.value, list):
            t = type(self.value[0])
            assert all([isinstance(v, t) for v in self.value])
            return list[t]
        elif isinstance(self.value, tuple):
            t = type(self.value[0])
            assert all([isinstance(v, t) for v in self.value])
            return tuple[t]
        raise ValueError(f"Cannot get generic type of {self.value}")


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
