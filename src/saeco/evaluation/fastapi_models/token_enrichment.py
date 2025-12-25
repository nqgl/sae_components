from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel

from saeco.evaluation.fastapi_models.EnrichmentSortBy import EnrichmentSortBy

from .filtered_query import FilterableQuery


class GeneInfo(BaseModel):
    category: str
    geneClass: str
    geneName: str
    displayBoth: bool


class TokenEnrichmentMode(str, Enum):
    doc = "doc"
    active = "active"
    max = "max"
    top = "top"


class TokenEnrichmentRequest(FilterableQuery):
    feature: int
    p: float | None = None
    k: int | None = None
    mode: TokenEnrichmentMode
    sort_by: EnrichmentSortBy = EnrichmentSortBy.counts
    num_top_tokens: int | None = 100


T = TypeVar("T")


class TokenEnrichmentResponseItem(BaseModel, Generic[T]):
    tokstr: str
    info: T
    token: int
    count: int
    normalized_count: float
    score: float


class TokenEnrichmentResponse(BaseModel, Generic[T]):
    results: list[TokenEnrichmentResponseItem[T]]
