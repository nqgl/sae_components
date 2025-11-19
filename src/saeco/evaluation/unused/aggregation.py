from collections.abc import Callable
from enum import Enum

from attr import define


class AggregationType(Enum):
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    ANY = "any"


@define
class Aggregation:
    agg: AggregationType | Callable = lambda x: x
    e = 2

    def aggregate(self, dim):
        return dim

    def multi_step(self, dim): ...
    def cumult(self, agg, add): ...
    def final_step(self, x, dim):
        seq_agg = self.agg
        docs_agg = "max"
        results = []
        if seq_agg == "count":
            c_agg = (x > 0).sum(dim=1)
        elif seq_agg == "max":
            c_agg = x.max(dim=1).values
        else:
            c_agg = getattr(x, seq_agg)(dim=1)
        if docs_agg == "max":
            c_agg = c_agg.max(dim=0).values
            results._max(c_agg)
        else:
            ...


import abc

abc.ABC  # comment


def afn(a, b, c):
    return a + b + c


e = True or False
d = {}
y = print

l = []

a = Aggregation(AggregationType.MEAN)
a.agg = 2
a.e = 5
v = a.e


def Afunc(x): ...


f1 = afn(1, 2, 3)
f2 = Afunc(2)
a.aggregate(2)
a = 2
b = a + 23


class Cls:
    def __init__(self): ...
    @classmethod
    def make_class(cls):
        return cls()


c = Cls()
c2 = Cls.make_class()
