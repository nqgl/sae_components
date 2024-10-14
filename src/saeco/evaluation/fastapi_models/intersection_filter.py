from pydantic import BaseModel


class GetIntersectionFilterKey(BaseModel):
    metadatas_values: dict[str, str | list[str] | int | list[int]] = {}
    initialzie_families: bool = False
