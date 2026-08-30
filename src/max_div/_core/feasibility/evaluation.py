"""This module evaluates selections and certificates over the packed constraint arrays.

It computes per-item scores at given prices, exact top-k selections, per-constraint counts,
violation aggregates, and the dual values that certify infeasibility.  Every function is pure
in its arrays; the pipelines that search for good prices live in sibling modules.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core._math import select_k_max

# soundness margin: g must exceed this before it counts as an infeasibility proof
G_POSITIVE_TOL = 1e-9


@numba.njit(cache=True)
def _item_scores(
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    lam_min: NDArray[np.float64],
    lam_max: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return per-item scores `s_j = sum of (lam_min_i - lam_max_i) over the constraints containing j`.

    Args:
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        lam_min: shortfall prices, one per constraint.
        lam_max: excess prices, one per constraint.
    """
    n = item_indptr.shape[0] - 1
    scores = np.zeros(n, dtype=np.float64)
    for j in range(n):
        acc = 0.0
        for e in range(item_indptr[j], item_indptr[j + 1]):
            i = item_cons[e]
            acc += lam_min[i] - lam_max[i]
        scores[j] = acc
    return scores


@numba.njit(cache=True)
def _top_k_items(scores: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """Return the indices of the k largest scores via O(n log k) heap selection.

    The result is an exact top-k with arbitrary ordering among ties — any tie-break is a valid
    inner maximizer, so the dual value stays sound.

    Args:
        scores: the per-item scores to select from.
        k: the selection size; `k >= len(scores)` returns every index.
    """
    if k >= scores.shape[0]:
        return np.arange(scores.shape[0], dtype=np.int64)
    return select_k_max(scores, np.int32(k)).astype(np.int64)


@numba.njit(cache=True)
def _selection_counts(
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    selection: NDArray[np.int64],
    m: int,
) -> NDArray[np.int64]:
    """Return per-constraint counts of selected members.

    Args:
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        selection: the selected item indices.
        m: the number of constraints.
    """
    counts = np.zeros(m, dtype=np.int64)
    for t in range(selection.shape[0]):
        j = selection[t]
        for e in range(item_indptr[j], item_indptr[j + 1]):
            counts[item_cons[e]] += 1
    return counts


@numba.njit(cache=True)
def _weighted_violation(
    counts: NDArray[np.int64],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> float:
    """Return the total weighted violation `sum of w_i * (shortfall_i + excess_i)`.

    Args:
        counts: per-constraint counts of selected members.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
    """
    v = 0.0
    for i in range(counts.shape[0]):
        if counts[i] < con_min[i]:
            v += weights[i] * (con_min[i] - counts[i])
        elif counts[i] > con_max[i]:
            v += weights[i] * (counts[i] - con_max[i])
    return v


@numba.njit(cache=True)
def _per_constraint_violation(
    counts: NDArray[np.int64], con_min: NDArray[np.int64], con_max: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Return how many items each constraint is short of, or over, its bounds.

    Unweighted item counts, so this stays readable as "constraint i misses by 3" whatever the
    weights are; `_weighted_violation` is the aggregate the verdicts are graded on.

    Args:
        counts: per-constraint counts of selected members.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
    """
    out = np.zeros(counts.shape[0], dtype=np.int64)
    for i in range(counts.shape[0]):
        if counts[i] < con_min[i]:
            out[i] = con_min[i] - counts[i]
        elif counts[i] > con_max[i]:
            out[i] = counts[i] - con_max[i]
    return out


@numba.njit(cache=True)
def _counts_feasible(counts: NDArray[np.int64], con_min: NDArray[np.int64], con_max: NDArray[np.int64]) -> bool:
    """Return whether every constraint count lies within its bounds.

    Args:
        counts: per-constraint counts of selected members.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
    """
    for i in range(counts.shape[0]):  # noqa: SIM110 — numba-compiled; the all(...) generator form is not supported
        if counts[i] < con_min[i] or counts[i] > con_max[i]:
            return False
    return True


@numba.njit(cache=True)
def _dual_value(
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    lam_min: NDArray[np.float64],
    lam_max: NDArray[np.float64],
    scores: NDArray[np.float64],
    selection: NDArray[np.int64],
) -> float:
    """Return the dual value `g = lam_min.min_counts - lam_max.max_counts - sum of selected scores`.

    Valid as a bound only when `selection` is an exact top-k of `scores`: `g` is the minimum of the
    priced penalty over all k-selections, and that minimum is attained at the exact top-k.  A
    non-maximizing selection overestimates `g` — the false-infeasibility-proof trap this module's
    tests pin explicitly.

    Args:
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        lam_min: shortfall prices, one per constraint.
        lam_max: excess prices, one per constraint.
        scores: per-item scores at these prices (`_item_scores`).
        selection: an exact top-k of `scores`.
    """
    g = 0.0
    for i in range(con_min.shape[0]):
        g += lam_min[i] * con_min[i] - lam_max[i] * con_max[i]
    for t in range(selection.shape[0]):
        g -= scores[selection[t]]
    return g


@numba.njit(cache=True)
def clamp_admissible(
    lam: NDArray[np.float64], w_lin: NDArray[np.float64], w_quad: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return a copy of the multipliers clamped into the admissible region.

    Admissible means nonnegative, and on purely linear constraints at most the linear weight —
    above it the dual value is minus infinity, so such prices certify nothing.  Clamping repairs
    floating-point drift in solver output; by weak duality the clamped pair certifies whatever
    `certified_bound` evaluates to, no matter how the input was produced.

    Args:
        lam: one multiplier per constraint (either the shortfall or the excess family).
        w_lin: per-constraint linear penalty weights.
        w_quad: per-constraint quadratic penalty weights.
    """
    out = lam.copy()
    for i in range(out.shape[0]):
        if out[i] < 0.0:
            out[i] = 0.0
        elif w_quad[i] == 0.0 and out[i] > w_lin[i]:
            out[i] = w_lin[i]
    return out


@numba.njit(cache=True)
def _price_cost(lam: float, w_lin: float, w_quad: float) -> float:
    """Return the price cost `psi(lam)` of one constraint: what pricing above the linear weight costs.

    Zero up to the linear weight; beyond it, finite only when a quadratic term exists (the caller
    guarantees admissibility, so the purely linear infinite branch never evaluates here).

    Args:
        lam: the constraint's (admissible) price.
        w_lin: the constraint's linear penalty weight.
        w_quad: the constraint's quadratic penalty weight.
    """
    excess = lam - w_lin
    if excess <= 0.0 or w_quad == 0.0:
        return 0.0
    return excess * excess / (4.0 * w_quad)


@numba.njit(cache=True)
def certified_bound(
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    w_lin: NDArray[np.float64],
    w_quad: NDArray[np.float64],
    lam_min: NDArray[np.float64],
    lam_max: NDArray[np.float64],
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    k: int,
) -> float:
    """Return the certified lower bound `g` on every selection's total penalty at admissible prices.

    The general-penalty dual value: the priced bounds minus the top-k of the item scores minus the
    per-constraint price costs.  For purely linear penalties the price costs vanish and this
    reduces to `_dual_value`.  A positive return proves infeasibility; the evaluation is sound for
    any admissible prices, however they were produced (clamp solver output with
    `clamp_admissible` first).

    Args:
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        w_lin: per-constraint linear penalty weights.
        w_quad: per-constraint quadratic penalty weights.
        lam_min: admissible shortfall prices.
        lam_max: admissible excess prices.
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        k: the selection size.
    """
    scores = _item_scores(item_indptr, item_cons, lam_min, lam_max)
    selection = _top_k_items(scores, k)
    g = _dual_value(con_min, con_max, lam_min, lam_max, scores, selection)
    for i in range(con_min.shape[0]):
        g -= _price_cost(lam_min[i], w_lin[i], w_quad[i])
        g -= _price_cost(lam_max[i], w_lin[i], w_quad[i])
    return g
