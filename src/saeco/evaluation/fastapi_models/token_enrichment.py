from pydantic import BaseModel, Field

from .FilterableQuery import FilterableQuery


class TokenEnrichmentRequest(FilterableQuery):
    feature: int
    p: float | None = None
    k: int | None = None


class TokenEnrichmentResultItem(BaseModel):
    label: str | int
    count: int
    normalized_count: float
    score: float | None


class TokenEnrichmentResult(BaseModel):
    active_position_results: list[TokenEnrichmentResultItem]
    active_document_results: list[TokenEnrichmentResultItem]
