from pydantic import BaseModel

from .Feature import Feature
from .filtered_query import FilterableQuery


class GetFamiliesRequest(FilterableQuery):
    # there might be some hyperparameters here but it might be just filter_id
    seq_level: bool = True


class FamilyLevel(BaseModel):
    level: int
    families: dict[int, "Family"]


class GetFamiliesResponse(BaseModel):
    levels: list[FamilyLevel]


class FamilyRef(BaseModel):
    level: int
    family_id: int


class ScoredFamilyRef(BaseModel):
    family: FamilyRef
    score: float


class ScoredFeature(BaseModel):
    feature: Feature
    score: float


class Family(BaseModel):
    level: int
    family_id: int
    label: str | None
    subfamilies: list[
        ScoredFamilyRef
    ]  # list of tuples of (family id, similarity score)
    # subfeatures: list[Feature]
    # or like this if the features have some sort of membership score:
    subfeatures: list[ScoredFeature]


# class Family:
#     level: int
#     family_id: int
#     label: str | None
#     # subfamilies: list[int]  # list of family ids
#     subfamilies: (
#         list[tuple["Family", float]] | None
#     )  # list of family ids with similarity score
#     # None if this is being not expanded. if it's actually empty,
#     # would be empty list
#     subfeatures: list[Feature] | None
#     # subfeatures: list[int] # or maybe list of feature ids?
#     # oh maybe we want to return the top n features,
#     # not all of them in case there's lots.
#     # what order do we want to return them in?


# class Family:
#     level: int
#     family_id: int
#     label: str | None
#     subfamilies: list[
#         tuple[int, float]
#     ]  # list of tuples of (family id, similarity score)
#     subfeatures: list[int]  # list of feature ids

#     # oh maybe we want to return the top n features,
#     # not all of them in case there's lots.
#     # what order do we want to return them in?


class GetFamilyDocumentActivityRequest(FilterableQuery):
    family_id: int


class FamilyTopActivatingExamplesQuery(FilterableQuery):
    families: list[FamilyRef]
    p: float | None = None
    k: int | None = None
    metadata_keys: list[str] = []
    return_str_docs: bool = False
    return_str_metadatas: bool = True


class SetFamilyLabelRequest(FilterableQuery):
    family: FamilyRef
    label: str


class TopFamilyOverlappingExamplesResponseDoc(BaseModel):
    doc: list[str] | list[int]
    metadatas: list[int | str]
    acts: list[list[float]] | None = None
    doc_index: int | None = None


class ActivationsOnDoc(BaseModel):
    document: list[str | int]
    metadatas: list[int | str]
    family_acts: list[list[float]]
    feature_acts: list[list[float]]


class ActivationsOnDocsRequest(FilterableQuery):
    document_ids: list[int]
    families: list[FamilyRef]
    feature_ids: list[int] = []
    return_str_docs: bool = True
    return_str_metadatas: bool = True
    metadata_keys: list[str] = []
