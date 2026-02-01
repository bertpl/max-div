from enum import StrEnum
from statistics import geometric_mean, mean


class TableAggregationType(StrEnum):
    MEAN = "mean"
    GEOMEAN = "geomean"
    SUM = "sum"

    def aggregate_values(self, values: list[int | float]) -> int | float:
        match self:
            case TableAggregationType.MEAN:
                return mean(values)
            case TableAggregationType.GEOMEAN:
                return geometric_mean(values)
            case TableAggregationType.SUM:
                return sum(values)
