from pydantic import BaseModel


class GetIntersectionFilterKey(BaseModel):
    metadatas_values: dict[str, str | list[str] | int | list[int]] = {}
    initialize_families: bool = False
