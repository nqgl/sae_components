from enum import Enum


class EnrichmentSortBy(str, Enum):
    counts = "count"
    normalized_count = "normalized_count"
    score = "score"
