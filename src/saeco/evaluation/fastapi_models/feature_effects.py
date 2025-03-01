from pydantic import BaseModel, Field

from .families_draft import FamilyRef

from .filtered_query import FilterableQuery


class LogitEffectsRequest(FilterableQuery):
    feature: int
    by_fwad: bool = True
    k: int = 100
    random_subset_n: int | None = None


class TopKFeatureEffects(BaseModel):
    pos_tokens: list[str]
    pos_values: list[float]
    neg_tokens: list[str]
    neg_values: list[float]
