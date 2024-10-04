from enum import Enum

from pydantic import BaseModel, Field

from .FilterableQuery import FilterableQuery


class TokenEnrichmentMode(str, Enum):
    doc = "doc"
    active = "active"
    max = "max"
    top = "top"


class TokenEnrichmentSortBy(str, Enum):
    counts = "count"
    normalized_count = "normalized_count"
    score = "score"


class TokenEnrichmentRequest(FilterableQuery):
    feature: int
    p: float | None = None
    k: int | None = None
    mode: TokenEnrichmentMode
    sort_by: TokenEnrichmentSortBy = TokenEnrichmentSortBy.counts
    num_top_tokens: int | None = 100


class TokenEnrichmentResponseItem(BaseModel):
    tokstr: str
    token: int
    count: int
    normalized_count: float
    score: float


class TokenEnrichmentResponse(BaseModel):
    results: list[TokenEnrichmentResponseItem]
