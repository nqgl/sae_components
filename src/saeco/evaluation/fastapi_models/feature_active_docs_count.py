from pydantic import BaseModel, Field

from .filtered_query import FilterableQuery


class FeatureActiveDocsRequest(FilterableQuery):
    feature: int


class FeatureActiveDocsResponse(BaseModel):
    num_active_docs: int
