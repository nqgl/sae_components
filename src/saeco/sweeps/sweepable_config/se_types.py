# from saeco.sweeps.sweepable_config.sweep_expressions import (
#     LITERAL_TYPE_COMBOS,
#     SweepExpression,
#     orlist,
# )

from typing import TypeAlias, Type, TypeVar, Union

from itertools import chain, combinations

from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression


T = TypeVar("T")


def any_literal_as_generic(t: Type[T]):
    return orlist([t[lit] for lit in LITERAL_TYPE_COMBOS])


def orlist(l: list):
    e = l[0]
    for i in l[1:]:
        e = e | i

    return e


# powerset from itertools recipes https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


LITERALS = [int, float, str, bool]
LITERAL_TYPE = int | float | str | bool


LITERAL_TYPE_COMBOS = [orlist(ss) for ss in powerset(LITERALS) if len(ss) > 0]

LiteralType: TypeAlias = int | float | str | bool
SweepExpressionAnyLiteral: TypeAlias = (
    SweepExpression[int]
    | SweepExpression[float]
    | SweepExpression[str]
    | SweepExpression[bool]
    | SweepExpression[Union[int, float]]
    | SweepExpression[Union[int, str]]
    | SweepExpression[Union[int, bool]]
    | SweepExpression[Union[float, str]]
    | SweepExpression[Union[float, bool]]
    | SweepExpression[Union[str, bool]]
    | SweepExpression[Union[int, float, str]]
    | SweepExpression[Union[int, float, bool]]
    | SweepExpression[Union[int, str, bool]]
    | SweepExpression[Union[float, str, bool]]
    | SweepExpression[Union[int, float, str, bool]]
)
