from pydantic import BaseModel

from .filtered_query import FilterableQuery


class FeatureActiveDocsRequest(FilterableQuery):
    feature: int


class FeatureActiveDocsResponse(BaseModel):
    num_active_docs: int
