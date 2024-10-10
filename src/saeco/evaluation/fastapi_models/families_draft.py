from pydantic import BaseModel

from .Feature import Feature

from .FilterableQuery import FilterableQuery


class GetFamiliesRequest(FilterableQuery):
    # there might be some hyperparameters here but it might be just filter_id
    seq_level: bool = True


class FamilyLevel(BaseModel):
    level: int
    families: dict[int, "Family"]


class GetFamiliesResponse(BaseModel):
    levels: list[FamilyLevel]


class Family(BaseModel):
    level: int
    family_id: int
    label: str | None
    subfamilies: list[
        tuple[int, float]
    ]  # list of tuples of (family id, similarity score)
    # subfeatures: list[Feature]
    # or like this if the features have some sort of membership score:
    subfeatures: list[Feature, float]


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
