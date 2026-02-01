from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from max_div.internal.benchmarking import BenchmarkResult
from max_div.internal.formatting import format_short_time_duration

from ._table_aggregation import TableAggregationType


# =================================================================================================
#  Base class
# =================================================================================================
class TableElement(ABC):
    """Element that can be placed in a table cell (regular rows, not headers)"""

    def __init__(self, supports_aggregation: bool = False):
        self._supports_aggregation = supports_aggregation

    @property
    def supports_aggregation(self) -> bool:
        return self._supports_aggregation

    @abstractmethod
    def to_mark_down(self) -> str:
        raise NotImplementedError()

    def to_plain_text(self) -> list[str]:
        return [self.to_mark_down()]

    @classmethod
    def aggregate(cls, elements: list[Self], agg_type: TableAggregationType) -> Self:
        raise NotImplementedError("Aggregation not supported for this TableElement type")

    def __str__(self):
        return self.to_mark_down()

    def __lt__(self, other):
        """Comparison operator, for table highlighting purposes."""
        return str(self) < str(other)

    def is_equalish(self, other):
        """To check if two TableElements are equal enough to be considered ties (lowest or highest) for highlighting."""
        return str(self).lower() == str(other).lower()


# =================================================================================================
#  Basic Elements
# =================================================================================================
class TableText(TableElement):
    def __init__(self, txt: str | list[str]):
        super().__init__(supports_aggregation=False)
        if isinstance(txt, str):
            txt = txt.splitlines()
        self.txt: list[str] = txt

    def to_mark_down(self) -> str:
        return "<br>".join(self.txt)

    def to_plain_text(self) -> list[str]:
        lines = self.txt.copy()
        for i in range(len(lines)):
            # strip each line of symmetrically leading & trailing asterisks (=Markdown-specific formatting)
            while lines[i] and lines[i][0] == "*" and lines[i][-1] == "*":
                lines[i] = lines[i][1:-1]
        return lines


class TablePercentage(TableElement):
    def __init__(self, frac: float, decimals: int = 1):
        super().__init__(supports_aggregation=True)
        self.frac = frac  # fraction between 0.0 and 1.0
        self.decimals = decimals  # number of decimals to display

    def to_mark_down(self) -> str:
        return f"{(self.frac * 100):.{self.decimals}f}%"

    def __lt__(self, other: TablePercentage):
        return self.frac < other.frac

    @classmethod
    def aggregate(cls, elements: list[TablePercentage], agg_type: TableAggregationType) -> TablePercentage:
        frac_values = [el.frac for el in elements]
        max_decimals = max(el.decimals for el in elements)
        return TablePercentage(
            frac=agg_type.aggregate_values(frac_values),
            decimals=max_decimals + 1 if agg_type != TableAggregationType.SUM else max_decimals,
        )


# =================================================================================================
#  Elements with uncertainties
# =================================================================================================
class _QuantiledTableElement(TableElement, ABC):
    def __init__(self, q_25: float, q_50: float, q_75: float):
        super().__init__(supports_aggregation=True)
        self.q_25 = q_25
        self.q_50 = q_50
        self.q_75 = q_75

    def __lt__(self, other: _QuantiledTableElement):
        return self.q_50 < other.q_50

    def is_equalish(self, other: _QuantiledTableElement):
        """True of both medians are in range of the other's 25-75 percentile."""
        return (self.q_25 <= other.q_50 <= self.q_75) and (other.q_25 <= self.q_50 <= other.q_75)

    @classmethod
    def _aggregate_quantiles(
        cls, elements: list[_QuantiledTableElement], agg_type: TableAggregationType
    ) -> tuple[float, float, float]:
        q_25_agg = agg_type.aggregate_values([el.q_25 for el in elements])
        q_50_agg = agg_type.aggregate_values([el.q_50 for el in elements])
        q_75_agg = agg_type.aggregate_values([el.q_75 for el in elements])
        return q_25_agg, q_50_agg, q_75_agg


class TableTimeElapsed(_QuantiledTableElement):
    def __init__(self, t_sec_q_25: float, t_sec_q_50: float, t_sec_q_75: float):
        super().__init__(q_25=t_sec_q_25, q_50=t_sec_q_50, q_75=t_sec_q_75)

    def to_mark_down(self) -> str:
        s_median = format_short_time_duration(dt_sec=self.q_50, right_aligned=True, spaced=True, long_units=True)
        s_perc = f"{50 * (self.q_75 - self.q_25) / self.q_50:.1f}%"
        return f"{s_median.strip()} ± {s_perc}"

    @classmethod
    def aggregate(cls, elements: list[TableTimeElapsed], agg_type: TableAggregationType) -> TableTimeElapsed:
        q_25_agg, q_50_agg, q_75_agg = cls._aggregate_quantiles(elements, agg_type)
        return TableTimeElapsed(t_sec_q_25=q_25_agg, t_sec_q_50=q_50_agg, t_sec_q_75=q_75_agg)

    @classmethod
    def from_benchmark_result(cls, result: BenchmarkResult) -> TableTimeElapsed:
        return TableTimeElapsed(
            t_sec_q_25=result.t_sec_q_25,
            t_sec_q_50=result.t_sec_q_50,
            t_sec_q_75=result.t_sec_q_75,
        )

    @classmethod
    def from_values(cls, t_values: list[float]) -> TableTimeElapsed:
        return TableTimeElapsed(
            t_sec_q_25=float(np.quantile(t_values, 0.25)),
            t_sec_q_50=float(np.quantile(t_values, 0.50)),
            t_sec_q_75=float(np.quantile(t_values, 0.75)),
        )


class TableValueWithUncertainty(_QuantiledTableElement):
    def __init__(self, value_q_25: float, value_q_50: float, value_q_75: float, decimals: int = 3):
        super().__init__(q_25=value_q_25, q_50=value_q_50, q_75=value_q_75)
        self.decimals = decimals

    def to_mark_down(self) -> str:
        s_median = f"{self.q_50:.{self.decimals}f}"
        s_perc = f"{50 * (self.q_75 - self.q_25) / self.q_50:.1f}%"
        return f"{s_median} ± {s_perc}"

    @classmethod
    def aggregate(
        cls, elements: list[TableValueWithUncertainty], agg_type: TableAggregationType
    ) -> TableValueWithUncertainty:
        q_25_agg, q_50_agg, q_75_agg = cls._aggregate_quantiles(elements, agg_type)
        max_decimals = max(el.decimals for el in elements)
        return TableValueWithUncertainty(
            value_q_25=q_25_agg,
            value_q_50=q_50_agg,
            value_q_75=q_75_agg,
            decimals=max_decimals,
        )

    @classmethod
    def from_values(cls, t_values: list[float], decimals: int = 3) -> TableValueWithUncertainty:
        return TableValueWithUncertainty(
            value_q_25=float(np.quantile(t_values, 0.25)),
            value_q_50=float(np.quantile(t_values, 0.50)),
            value_q_75=float(np.quantile(t_values, 0.75)),
            decimals=decimals,
        )
