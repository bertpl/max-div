from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from max_div.internal.benchmarking import BenchmarkResult


# =================================================================================================
#  Percentage
# =================================================================================================
@dataclass
class Percentage:
    frac: float  # fraction between 0.0 and 1.0
    decimals: int = 1  # number of decimals to display

    def __str__(self):
        return f"{(self.frac * 100):.{self.decimals}f}%"


# =================================================================================================
#  NumberWithUncertainty
# =================================================================================================
@dataclass(frozen=True)
class NumberWithUncertainty:
    value_q_25: float
    value_q_50: float
    value_q_75: float
    decimals: int = 3

    @property
    def value_str(self) -> str:
        return f"{self.value_q_50:.{self.decimals}f}"

    @property
    def value_with_uncertainty_str(self) -> str:
        s_median = self.value_str
        s_perc = f"{50 * (self.value_q_75 - self.value_q_25) / self.value_q_50:.1f}%"
        return f"{s_median} ± {s_perc}"

    def __str__(self) -> str:
        return self.value_with_uncertainty_str

    @classmethod
    def from_list(cls, lst: list[float], decimals: int = 3) -> NumberWithUncertainty:
        """
        Create a NumberWithUncertainty from a list of measured values.

        :param lst: List of measured values
        :param decimals: Number of decimals to display
        :return: NumberWithUncertainty with computed q25, q50, q75
        """
        q25, q50, q75 = np.quantile(lst, [0.25, 0.50, 0.75])
        return NumberWithUncertainty(
            value_q_25=float(q25),
            value_q_50=float(q50),
            value_q_75=float(q75),
            decimals=decimals,
        )

    @classmethod
    def aggregate(
        cls, results: list[NumberWithUncertainty], method: Literal["mean", "geomean", "sum"]
    ) -> NumberWithUncertainty:
        """
        Aggregate multiple NumberWithUncertainty objects into a single result,
        by aggregating q25, q50, 75 values separately.

        :param results: List of NumberWithUncertainty objects to aggregate
        :param method: Aggregation method - "mean", "geomean" (geometric mean), or "sum"
        :return: Aggregated NumberWithUncertainty
        """
        if not results:
            raise ValueError("Cannot aggregate empty list of results")

        # Collect all quantile values
        q25_values = [r.value_q_25 for r in results]
        q50_values = [r.value_q_50 for r in results]
        q75_values = [r.value_q_75 for r in results]

        # Apply the aggregation method
        if method == "mean":
            agg_q25 = np.mean(q25_values)
            agg_q50 = np.mean(q50_values)
            agg_q75 = np.mean(q75_values)
        elif method == "geomean":
            agg_q25 = np.exp(np.mean(np.log(q25_values)))
            agg_q50 = np.exp(np.mean(np.log(q50_values)))
            agg_q75 = np.exp(np.mean(np.log(q75_values)))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return NumberWithUncertainty(
            value_q_25=agg_q25,
            value_q_50=agg_q50,
            value_q_75=agg_q75,
            decimals=max([r.decimals for r in results]),
        )


# =================================================================================================
#  Aggregate types
# =================================================================================================
CellContent = str | BenchmarkResult | Percentage | NumberWithUncertainty
