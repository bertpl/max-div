"""`DistanceStorageType` names the user's choice of distance backend."""

from enum import StrEnum


class DistanceStorageType(StrEnum):
    """A `DistanceStorageType` names how the solver stores pairwise distances during search.

    `AUTO` (the default) lets max-div decide; `select_distance_storage_type` states the policy.  The
    resolved backend is reported in the solution summary.  Pinning a backend overrides the policy —
    `LAZY` requires vectors, so it is unavailable for distance-input problems.  A condensed distance
    input is expanded to the full matrix, at twice its memory.
    """

    AUTO = "auto"
    FULL_MATRIX = "full_matrix"
    LAZY = "lazy"
