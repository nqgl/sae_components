from pydantic import BaseModel


class Feature(BaseModel):
    feature_id: int
    label: str | None = None
