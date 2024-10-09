from pydantic import BaseModel


class Feature(BaseModel):
    feature_id: int
    label: str | None = None


class FilterConditionedResponse(BaseModel): ...


class FiltFeature(FilterConditionedResponse):
    feature_id: int
    label: str | None = None
