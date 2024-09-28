from enum import Enum
from typing import Callable

from attr import define


class AggregationType(Enum):
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    ANY = "any"


@define
class Aggregation:
    agg: AggregationType | Callable

    def aggregate(self, dim): ...

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
