"""Shared on-disk schema constants; this module intentionally has no ML dependency."""

FEATURE_NAMES = (
    "ratio_to_entry",
    "ratio_to_first",
    "search_progress",
    "rank_norm",
    "persistence_rate",
    "avg_neigh_dist",
)
FEATURE_COUNT = len(FEATURE_NAMES)


def query_group_name(query_id: int) -> str:
    return f"query_{query_id:05d}"
