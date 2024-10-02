from pydantic import BaseModel, Field


class TopActivatingExamplesResult(BaseModel):
    docs: list[list[str]] | list[list[int]]
    metadatas: list[list[int]]
    acts: list[list[float]] | None = None
    doc_indices: list[int] | None = None


class TopActivationResultEntry(BaseModel):
    docs: list[str] | list[int]
    metadatas: list[int]
    acts: list[float] | None = None
    doc_indices: int | None = None


class TopActivatingExamplesResult(BaseModel):
    entries: list[TopActivationResultEntry]


class TopActivatingExamplesQuery(BaseModel):
    feature: int
    p: float | None = None
    k: int | None = None
    metadata_keys: list[str] = []
    return_str_docs: bool = False
