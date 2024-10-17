from pydantic import BaseModel, Field

from .filtered_query import FilterableQuery


class FeatureLogitEffectsRequest(FilterableQuery):
    feature: int
    by_fwad: bool = True
    k: int = 100
    random_subset_n: int = None


class TopKFeatureEffects(BaseModel):
    tokens: list[str]
    values: list[float]
