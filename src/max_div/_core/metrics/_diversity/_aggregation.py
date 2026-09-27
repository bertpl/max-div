"""A hybrid aggregation combines the values of a hybrid objective's terms into one value, with one weight per term.

What a weight means depends on the aggregation, so each aggregation holds its own weights. Below,
s_t is term t's value and w_t its weight:

- `GeometricMeanAggregation` — a weight is its term's exponent: (prod_t s_t ** w_t) ** (1 / sum(w)).
- `ArithmeticMeanAggregation` — a weight multiplies its term's value: sum_t w_t * s_t / sum(w).

Both normalize by the weight sum, so equal weights give the plain mean.
"""

from __future__ import annotations

import math
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np

from max_div._core._math.geomean import weighted_geomean_per_row_f32
from max_div._core._math.mean import weighted_mean_per_row_f32

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


# =================================================================================================
#  HybridAggregation
# =================================================================================================
@dataclass(frozen=True)
class HybridAggregation(ABC):
    """A hybrid aggregation combines a hybrid objective's term values into one value, with one weight per term."""

    # `name` is the aggregation's short name: it starts the hybrid's label (`geomean(...)`), and appending
    # `_of` gives the public factory method (`geomean_of`).
    name: ClassVar[str]

    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        """Store the weights as plain floats, so `repr` shows an int `2` and a numpy `2` both as `2.0`.

        Raises:
            ValueError: If a weight is not a positive, finite number; a bool is rejected although it is a
                `numbers.Real`.
        """
        for weight in self.weights:
            if (
                isinstance(weight, bool)
                or not isinstance(weight, numbers.Real)
                or not (math.isfinite(weight) and weight > 0)
            ):
                raise ValueError(
                    f"Each weight of a hybrid diversity metric must be a positive, finite number; got {weight!r}."
                )
        object.__setattr__(self, "weights", tuple(float(weight) for weight in self.weights))

    @classmethod
    def with_unit_weights(cls, n_terms: int) -> Self:
        """Return this aggregation with a weight of 1 for each of `n_terms` terms."""
        return cls((1.0,) * n_terms)

    @classmethod
    def from_weights(cls, weights: Sequence[float] | None, n_terms: int) -> Self:
        """Return this aggregation with `weights`, or with a weight of 1 per term when `weights` is `None`."""
        if weights is None:
            return cls.with_unit_weights(n_terms)
        else:
            return cls(tuple(weights))

    # --------------------------------------------------------------------------
    #  Properties
    # --------------------------------------------------------------------------
    @property
    def has_non_unit_weights(self) -> bool:
        """Return whether any weight differs from 1."""
        return any(weight != 1.0 for weight in self.weights)

    @cached_property
    def weights_f32(self) -> NDArray[np.float32]:
        """Return the weights as a float32 array, the input dtype of the per-row mean functions."""
        return np.array(self.weights, dtype=np.float32)

    # --------------------------------------------------------------------------
    #  Validation and labels
    # --------------------------------------------------------------------------
    def validate_term_count(self, n_terms: int) -> None:
        """Check that this aggregation holds one weight per term.

        Raises:
            ValueError: If the number of weights differs from `n_terms`.
        """
        if len(self.weights) != n_terms:
            raise ValueError(
                f"A hybrid diversity metric needs one weight per term; got {len(self.weights)} weights "
                f"for {n_terms} terms."
            )

    def format_label(self, term_labels: Sequence[str]) -> str:
        """Return e.g. `geomean(A, B)`, or `geomean(A, B; weights 2, 1)` when the weights are not all 1.

        Each weight is shown to 4 significant digits.
        """
        weights_suffix = (
            f"; weights {', '.join(f'{weight:.4g}' for weight in self.weights)}" if self.has_non_unit_weights else ""
        )
        return f"{self.name}({', '.join(term_labels)}{weights_suffix})"

    # --------------------------------------------------------------------------
    #  Aggregation
    # --------------------------------------------------------------------------
    def aggregate_rows(self, rows: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the aggregate of each row of `rows`, as a fresh float32 array.

        Args:
            rows: a C-contiguous float32 (n_rows, n_terms) array, one column per term in term order.
        """
        aggregated = np.empty(rows.shape[0], dtype=np.float32)
        self._aggregate_rows_into(rows, aggregated)
        return aggregated

    def aggregate_scores(self, term_scores: NDArray[np.float32]) -> float:
        """Return the aggregate of `term_scores`, a contiguous float32 array of the terms' scores in term order.

        This is `aggregate_rows` over the single row of scores, so a hybrid's score and its per-item
        contributions combine their terms by the same arithmetic.
        """
        return float(self.aggregate_rows(term_scores.reshape(1, -1))[0])

    @abstractmethod
    def _aggregate_rows_into(self, rows: NDArray[np.float32], out: NDArray[np.float32]) -> None:
        """Write the aggregate of each row of `rows` into `out`, which has one entry per row."""


# =================================================================================================
#  Concrete aggregations
# =================================================================================================
@dataclass(frozen=True)
class GeometricMeanAggregation(HybridAggregation):
    """The weighted geometric mean uses each weight as its term's exponent, normalized by the weight sum.

    The mean is zero as soon as one term is zero. Scaling a term by a constant scales the mean by a
    constant as well, so rescaling a term does not change which selection scores highest.
    """

    name: ClassVar[str] = "geomean"

    def _aggregate_rows_into(self, rows: NDArray[np.float32], out: NDArray[np.float32]) -> None:
        """Write each row's weighted geometric mean into `out`."""
        weighted_geomean_per_row_f32(rows, self.weights_f32, out)


@dataclass(frozen=True)
class ArithmeticMeanAggregation(HybridAggregation):
    """The weighted arithmetic mean sums each term's value times its weight and divides by the weight sum."""

    name: ClassVar[str] = "mean"

    def _aggregate_rows_into(self, rows: NDArray[np.float32], out: NDArray[np.float32]) -> None:
        """Write each row's weighted arithmetic mean into `out`."""
        weighted_mean_per_row_f32(rows, self.weights_f32, out)
