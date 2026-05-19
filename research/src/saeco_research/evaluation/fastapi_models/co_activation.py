from pydantic import BaseModel, Field

from .filtered_query import FilterableQuery


class CoActivatingFeature(BaseModel):
    feature_id: int
    coactivation_level: float  # cosine similarity value or count or whatever


class CoActivationResponse(BaseModel):
    results: list[CoActivatingFeature] = Field(
        description="A list of co-activating features"
    )


class CoActivationRequest(FilterableQuery):
    feature_id: int
    n: int = Field(description="Number of top co-activating features to return")
    mode: str = "seq"
