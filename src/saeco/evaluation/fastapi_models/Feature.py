from pydantic import BaseModel


class LabeledFeature(BaseModel):
    feature_id: int
    label: str | None = None
