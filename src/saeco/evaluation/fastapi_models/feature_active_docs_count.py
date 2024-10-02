from pydantic import BaseModel, Field

from .FilterableQuery import FilterableQuery


class FeatureActiveDocsRequest(FilterableQuery):
    feature: int


class FeatureActiveDocsResponse(BaseModel):
    num_active_docs: int
