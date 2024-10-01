from pydantic import BaseModel, Field


class MetadataLabelEnrichment(BaseModel):
    count: int | None
    score: float | None
    label: str | int
    act_sum: float | None = None


# class MetadataEnrichmentResult(BaseModel):
#     metadata_key: str
#     results: list[MetadataLabelEnrichment]


class MetadataEnrichmentResults(BaseModel):
    # results: list[MetadataEnrichmentResult] # could be a dict instead
    results: dict[str, list[MetadataLabelEnrichment]] = Field(
        description="A dictionary of metadata keys to ordered lists of MetadataLabelEnrichment objects"
    )


class TopActivatingExamplesQuery(BaseModel):
    feature: int
    p: float | None = None
    k: int | None = None
    metadata_keys: list[str] = []
    return_str_docs: bool = False
    # return_acts_sparse: bool = False,


class TopActivatingExamplesResult(BaseModel):
    docs: list[list[str]] | list[list[int]]
    metadatas: list[list[int]]
    acts: list[list[float]] | None = None
    sparse_acts: dict[list[list[int]], float] | None = None
    doc_indices: list[int] | None = None
