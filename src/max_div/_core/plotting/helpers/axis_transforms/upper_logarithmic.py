"""This module defines the upper-logarithmic axis transform, for values that approach a limit from below."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .base import AxisTransform, as_float_or_array

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class UpperLogTransform(AxisTransform):
    """A logarithmic transform of the distance to a limit above the data, `log(x_ref - x)`.

    A score that saturates stays within a narrow band just below its final value for most of a solve,
    and a linear axis draws that band as a nearly flat line. This transform stretches that band: the
    axis coordinate is

        to_axis(x)   = c0 + c1 * log(x_ref - x)
        from_axis(x') = x_ref - exp((x' - c0) / c1)

    so equal ratios of the remaining distance to `x_ref` take equal room on the axis. The constructor
    takes the three parameters as they are; `from_values` fits them to data so that the data's range
    maps onto [0, 1] as uniformly as possible.
    """

    # --------------------------------------------------------------------------
    #  Constructor
    # --------------------------------------------------------------------------
    def __init__(self, c0: float, c1: float, x_ref: float) -> None:
        """Store the offset `c0`, the scale `c1` and the limit `x_ref` that the data approaches."""
        self.c0 = c0
        self.c1 = c1
        self.x_ref = x_ref

    # --------------------------------------------------------------------------
    #  Main API
    # --------------------------------------------------------------------------
    def to_axis(self, x: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return `c0 + c1 * log(x_ref - x)`."""
        return as_float_or_array(self.c0 + self.c1 * np.log(self.x_ref - np.asarray(x, dtype=np.float64)))

    def from_axis(self, x_axis: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return `x_ref - exp((x_axis - c0) / c1)`."""
        return as_float_or_array(self.x_ref - np.exp((np.asarray(x_axis, dtype=np.float64) - self.c0) / self.c1))

    def ticks_and_labels(
        self, n_max: int = 20, should_add_missing_ticks: bool = False
    ) -> tuple[list[float], list[str]]:
        """Return tick positions in data units, about evenly spaced on the axis, with labels of minimal precision.

        The ticks are chosen from a grid that is uniform on the axis, each rounded to the fewest digits
        that still separate it from its neighbors, so the labels read `0.5, 0.7, 0.79, 0.799, ...` and never
        repeat. The grid is made finer, starting from `n_max` points, until rounding leaves as close to
        `n_max` distinct labels as possible, never more.

        Args:
            n_max: The largest number of labeled ticks to return.
            should_add_missing_ticks: When the precision jumps between two neighbors, e.g. from `1.414`
                to `1.4144`, also return the unlabeled ticks in between (`1.4141`, `1.4142`, `1.4143`)
                so the axis shows a regular grid there. Their labels are empty strings.

        Returns:
            `(ticks, labels)`: tick positions in data units, ascending, and one label per tick.
        """
        # --- search the tick count that gives closest to n_max distinct labels
        ticks: list[float] = []
        labels: list[str] = []
        for n_max_internal in range(n_max, 10 * n_max):
            candidate_ticks, candidate_labels = self._rounded_ticks_and_labels(n_max_internal)
            if len(candidate_ticks) == n_max:
                ticks, labels = candidate_ticks, candidate_labels
                break
            if len(candidate_ticks) <= n_max:
                ticks, labels = candidate_ticks, candidate_labels
            elif len(candidate_ticks) >= 2 * n_max:
                break

        # --- fill precision jumps with unlabeled ticks
        if should_add_missing_ticks and len(ticks) >= 2:
            ticks, labels = _add_ticks_at_precision_jumps(ticks, labels)

        return ticks, labels

    # --------------------------------------------------------------------------
    #  Factory
    # --------------------------------------------------------------------------
    @classmethod
    def from_values(cls, x: ArrayLike, n_candidates: int = 100) -> UpperLogTransform:
        """Fit the transform to data so that its range maps onto [0, 1] as uniformly as possible.

        The limit `x_ref` is chosen among candidates above `max(x)`, each the maximum plus its distance
        to one quantile of the data, and for each candidate `c0` and `c1` are set so that `min(x)` maps
        to 0 and `max(x)` to 1. The candidate whose transformed data is closest to uniformly spread over
        [0, 1] is chosen.

        Constant data gets a transform with `x_ref` one above the value and the value itself drawn at 0.5.

        Args:
            x: The data values the axis has to show.
            n_candidates: The number of `x_ref` candidates to try; more is slower and finer.
        """
        # --- prepare ----------------------------
        x_sorted = np.sort(np.asarray(x, dtype=np.float64))
        x_min, x_max = float(x_sorted[0]), float(x_sorted[-1])
        if x_min == x_max:
            # c1 < 0 keeps the axis increasing in x, as it is under every fitted transform
            return UpperLogTransform(c0=0.5, c1=-1.0, x_ref=x_max + 1.0)

        # --- x_ref candidates: quantiles mirrored around the maximum
        q_values = 1.0 - np.linspace(1 / n_candidates, 1.0, n_candidates) ** 2
        x_ref_candidates = sorted({x_max + (x_max - float(np.quantile(x_sorted, q))) for q in q_values})
        x_ref_candidates = [x_ref for x_ref in x_ref_candidates if x_ref > x_max]

        # --- keep the candidate with the most uniform transformed data
        best_x_ref, best_c0, best_c1 = 0.0, 0.0, 0.0
        best_uniformity = -math.inf
        uniform_quantiles = np.linspace(0.0, 1.0, 101)
        for x_ref in x_ref_candidates:
            c1 = 1.0 / (math.log(x_ref - x_max) - math.log(x_ref - x_min))
            c0 = -c1 * math.log(x_ref - x_min)
            x_axis = UpperLogTransform(c0=c0, c1=c1, x_ref=x_ref).to_axis(x_sorted)
            # uniform data has its q-quantile at q; the sum of squared deviations from that measures the non-uniformity
            uniformity = 1.0 - float(np.sum((uniform_quantiles - np.quantile(x_axis, uniform_quantiles)) ** 2))
            if uniformity > best_uniformity:
                best_uniformity, best_x_ref, best_c0, best_c1 = uniformity, x_ref, c0, c1

        return UpperLogTransform(c0=float(best_c0), c1=float(best_c1), x_ref=float(best_x_ref))

    # --------------------------------------------------------------------------
    #  Internal
    # --------------------------------------------------------------------------
    def _rounded_ticks_and_labels(self, n_ticks: int) -> tuple[list[float], list[str]]:
        """Return `n_ticks` axis-uniform positions rounded to the fewest digits that separate neighbors, de-duplicated.

        Rounding merges neighbors that fall within each other's precision, so fewer than `n_ticks`
        positions usually come back; `ticks_and_labels` raises `n_ticks` until the count fits.
        """
        # --- positions uniform on the axis, in data units
        n_ticks = max(2, n_ticks)
        step_axis = 1.0 / (n_ticks - 1)
        positions_axis = np.linspace(0.0, 1.0, n_ticks)
        positions = np.asarray(self.from_axis(positions_axis))

        # --- round each position to the precision of the step around it
        labels: list[str] = []
        for x, x_axis in zip(positions, positions_axis, strict=True):
            data_step = float(self.from_axis(x_axis + 0.5 * step_axis)) - float(
                self.from_axis(x_axis - 0.5 * step_axis)
            )
            n_digits = max(1, -math.floor(math.log10(data_step))) - 1
            x_rounded = int(x / 10**-n_digits) * 10**-n_digits
            labels.append(f"{x_rounded:.{n_digits}f}")

        # --- de-duplicate, sort, drop what rounding pushed off the axis
        labels = sorted(set(labels), key=float)
        labels = [label for label in labels if -0.05 < float(self.to_axis(float(label))) < 1.05]
        if not labels:
            # coarse rounding can push every position off the axis; a finer grid is tried next
            return [], []

        # --- drop a trailing zero where the label's precision is a step up from its predecessor
        labels[0] = _without_trailing_zero(labels[0])
        for i in range(1, len(labels) - 1):
            if len(labels[i]) > len(labels[i - 1]):
                labels[i] = _without_trailing_zero(labels[i])

        return [float(label) for label in labels], labels


# ==================================================================================================
#  Helpers
# ==================================================================================================
def _without_trailing_zero(label: str) -> str:
    """Return `label` without one trailing zero after the decimal point; `1.0` becomes `1`."""
    if "." not in label:
        return label
    elif label.endswith(".0"):
        return label[:-2]
    elif label.endswith("0"):
        return label[:-1]
    else:
        return label


def _add_ticks_at_precision_jumps(ticks: list[float], labels: list[str]) -> tuple[list[float], list[str]]:
    """Insert unlabeled ticks between two neighbors whose label precision jumps, at the finer precision's step."""
    ticks_out: list[float] = []
    labels_out: list[str] = []
    for i, (tick, label) in enumerate(zip(ticks, labels, strict=True)):
        ticks_out.append(tick)
        labels_out.append(label)
        if i == len(ticks) - 1:
            break
        next_label = labels[i + 1]
        if len(next_label) > len(label) and "." in next_label:
            step = 10 ** -len(next_label.split(".")[1])
            intermediate = tick + step
            while intermediate < ticks[i + 1] - step / 2:
                ticks_out.append(intermediate)
                labels_out.append("")
                intermediate += step
    return ticks_out, labels_out
