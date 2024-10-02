from pydantic import BaseModel, Field

from .FilterableQuery import FilterableQuery


class MetadataEnrichmentRequest(FilterableQuery):
    feature: int
    metadata_keys: list[str] = Field
    p: float | None = None
    k: int | None = None
    str_label: bool = False


class MetadataEnrichmentLabelResult(BaseModel):
    label: str | int
    count: int
    proportion: float
    normalized_count: float
    score: float | None


class MetadataEnrichmentResponse(BaseModel):
    results: dict[str, list[MetadataEnrichmentLabelResult]] = Field(
        description="A dictionary of metadata keys to ordered lists of MetadataLabelEnrichment objects"
    )
