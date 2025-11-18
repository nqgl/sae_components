from types import UnionType
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression

LITERALS = [int, float, str, bool]


def common_type(objs: list["SweepExpression"]):
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


def shared_type[T](it: Iterable[T]) -> type[T] | UnionType:
    l = list(it)
    t = type(l[0])
    for v in l[1:]:
        t |= type(v)
    return t


def convert_other(other):
    from saeco.sweeps.sweepable_config.sweep_expressions import Val
    from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression

    if isinstance(other, SweepExpression):
        return other
    if any([isinstance(other, l) for l in LITERALS]):
        return Val[type(other)](value=other)
    # if isinstance(other, SweepVar):
    #     return Var(name=other.name, sweep_var=other)
    raise ValueError(f"Cannot convert {other} to SweepExpressionNode")
