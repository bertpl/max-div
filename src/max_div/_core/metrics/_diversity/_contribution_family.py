from enum import StrEnum


class DiversityContributionFamily(StrEnum):
    """The members name the per-point diversity-contribution families that diversity metrics consume.

    Members
    -------

        - SEPARATION:     contribution = distance to the nearest selected item
        - MEAN_DISTANCE:  contribution = mean distance to the selected items
    """

    SEPARATION = "SEPARATION"
    MEAN_DISTANCE = "MEAN_DISTANCE"
