from pydantic import BaseModel, Field


class MetadataEnrichmentRequest(BaseModel):
    feature: int
    metadata_keys: list[str] = Field
    p: float | None = None
    k: int | None = None
    str_label: bool = False
    normalize_metadata_frequencies: bool = ...


class MetadataEnrichmentLabelResult(BaseModel):
    label: str | int
    count: int | float | None
    score: float | None
    act_sum: float | None = None


class MetadataEnrichmentResponse(BaseModel):
    results: dict[str, list[MetadataEnrichmentLabelResult]] = Field(
        description="A dictionary of metadata keys to ordered lists of MetadataLabelEnrichment objects"
    )
