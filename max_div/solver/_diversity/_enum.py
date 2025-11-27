from enum import StrEnum


class DiversityMetric(StrEnum):
    """
    Enum for different diversity metrics.

    Definitions
    -----------

        - SEPARATION of a vector is the distance to its nearest neighbor in the selection.

    Members
    -------

        - MIN_SEPARATION:       Minimum separation of all selected vectors
        - MEAN_SEPARATION:      Arithmetic mean separation of all selected vectors
        - GEOMEAN_SEPARATION:   Geometric mean separation of all selected vectors

    """

    MIN_SEPARATION = "MIN_SEPARATION"
    MEAN_SEPARATION = "MEAN_SEPARATION"
    GEOMEAN_SEPARATION = "GEOMEAN_SEPARATION"
