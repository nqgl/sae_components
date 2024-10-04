from pydantic import BaseModel, Field

from .FilterableQuery import FilterableQuery


class TopActivationResultEntry(BaseModel):
    doc: list[str] | list[int]
    metadatas: list[int | str]
    acts: list[float] | None = None
    doc_index: int | None = None


class TopActivatingExamplesResult(BaseModel):
    entries: list[TopActivationResultEntry]


class TopActivatingExamplesQuery(FilterableQuery):
    feature: int
    p: float | None = None
    k: int | None = None
    metadata_keys: list[str] = []
    return_str_docs: bool = False
    return_str_metadatas: bool = True
