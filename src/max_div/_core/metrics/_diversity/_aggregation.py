"""A hybrid aggregation combines the values of a hybrid objective's terms into one value, with one weight per term.

What a weight means depends on the aggregation, so each aggregation holds its own weights:

- `GeometricMeanAggregation` — a weight is its term's exponent: (prod_t s_t ** w_t) ** (1 / sum(w)).
- `ArithmeticMeanAggregation` — a weight is its term's weight in the mean: sum_t w_t * s_t / sum(w).

Both normalize by the weight sum, so equal weights give the plain mean. An aggregation combines rows:
a matrix with one column per term gives one value per row. The per-item contribution is that
combination over a row per item; the hybrid's score is the same combination over the single row of
term scores.
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
    """How a hybrid objective combines its terms' values, with one positive, finite weight per term."""

    # the aggregation's short name: the label's prefix, and the public factory's name without `_of`
    name: ClassVar[str]

    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        """Store the weights as floats, rejecting any weight that is not a positive, finite number.

        Raises:
            ValueError: If a weight is not a positive, finite number.
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
    def with_equal_weights(cls, n_terms: int) -> Self:
        """Return this aggregation with a weight of 1 for each of `n_terms` terms."""
        return cls((1.0,) * n_terms)

    # --------------------------------------------------------------------------
    #  Properties
    # --------------------------------------------------------------------------
    @property
    def is_weighted(self) -> bool:
        """Return whether any weight differs from 1."""
        return any(weight != 1.0 for weight in self.weights)

    @cached_property
    def weights_f32(self) -> NDArray[np.float32]:
        """Return the weights as a float32 array, the form the row combinations compute with."""
        return np.array(self.weights, dtype=np.float32)

    # --------------------------------------------------------------------------
    #  Validation and labels
    # --------------------------------------------------------------------------
    def check_term_count(self, n_terms: int) -> None:
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
        weights = f"; weights {', '.join(f'{weight:.4g}' for weight in self.weights)}" if self.is_weighted else ""
        return f"{self.name}({', '.join(term_labels)}{weights})"

    # --------------------------------------------------------------------------
    #  Combination
    # --------------------------------------------------------------------------
    @abstractmethod
    def aggregate_rows(self, rows: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the combination of each row of `rows`, as a fresh float32 array.

        Args:
            rows: a C-contiguous float32 (n_rows, n_terms) array, one column per term in term order.
        """

    def aggregate_scores(self, term_scores: NDArray[np.float32]) -> float:
        """Return the combination of the terms' scores, a contiguous float32 array in term order.

        This is `aggregate_rows` over the single row of scores, so a hybrid's score and its per-item
        contributions combine their terms by the same arithmetic.
        """
        return float(self.aggregate_rows(term_scores.reshape(1, -1))[0])


# =================================================================================================
#  Concrete aggregations
# =================================================================================================
@dataclass(frozen=True)
class GeometricMeanAggregation(HybridAggregation):
    """The weighted geometric mean: each weight is its term's exponent, normalized by the weight sum.

    The mean is zero as soon as one term is zero. Scaling a term by a constant scales the mean by a
    constant as well, so it does not change which selection scores highest.
    """

    name: ClassVar[str] = "geomean"

    def aggregate_rows(self, rows: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return each row's weighted geometric mean."""
        aggregated = np.empty(rows.shape[0], dtype=np.float32)
        weighted_geomean_per_row_f32(rows, self.weights_f32, aggregated)
        return aggregated


@dataclass(frozen=True)
class ArithmeticMeanAggregation(HybridAggregation):
    """The weighted arithmetic mean: each term's value times its weight, summed and divided by the weight sum."""

    name: ClassVar[str] = "mean"

    def aggregate_rows(self, rows: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return each row's weighted arithmetic mean."""
        aggregated = np.empty(rows.shape[0], dtype=np.float32)
        weighted_mean_per_row_f32(rows, self.weights_f32, aggregated)
        return aggregated
