"""Randomized rounding turns the relaxation's fractional marginals into concrete selections.

The marginals are read as per-item inclusion probabilities: a draw selects exactly `k` items, item
`j` with probability `marginals[j]`, so every constraint's count is correct in expectation and
individual draws deviate only by sampling fluctuation.  Different draws land in different places —
the property that makes rounding-based initializations vary with the seed.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core._random import rand_float64

from .evaluation import _selection_counts, _weighted_violation
from .repair import _repair_selection


@numba.njit(cache=True)
def systematic_sample(marginals: NDArray[np.float64], k: int, rng_state: NDArray[np.uint64]) -> NDArray[np.int64]:
    """Draw a selection of exactly `k` items with per-item inclusion probabilities `marginals`.

    Systematic sampling: traverse the items in a random order, accumulate the probabilities, and
    select an item exactly when its accumulation interval contains a point of a randomly shifted
    unit-spaced grid.  The intervals partition `[0, k)` (the marginals sum to `k`), which contains
    exactly `k` grid points — so exactly `k` items are selected — and an interval of length at
    most 1 contains a grid point with probability equal to its length, which is the item's
    marginal.  A guard settles the rare boundary case where floating-point accumulation drifts a
    grid point across an interval edge.

    Args:
        marginals: per-item inclusion probabilities in [0, 1], summing to `k`.
        k: the selection size.
        rng_state: xoroshiro128+ state, advanced in place.
    """
    n = marginals.shape[0]

    # random traversal order (Fisher-Yates)
    order = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = np.int64(rand_float64(rng_state) * (i + 1))
        order[i], order[j] = order[j], order[i]

    # walk the shifted unit grid through the cumulative probabilities
    shift = rand_float64(rng_state)
    selection = np.empty(k, dtype=np.int64)
    n_selected = 0
    cumulative = 0.0
    prev_cell = np.int64(np.floor(-shift))
    for t in range(n):
        cumulative += marginals[order[t]]
        cell = np.int64(np.floor(cumulative - shift))
        if cell > prev_cell and n_selected < k:
            selection[n_selected] = order[t]
            n_selected += 1
        prev_cell = cell

    # float-drift guard: top up from the traversal order's largest-marginal leftovers
    if n_selected < k:
        _top_up_selection(marginals, selection, n_selected, k)
    return selection


@numba.njit(cache=True)
def _top_up_selection(marginals: NDArray[np.float64], selection: NDArray[np.int64], n_selected: int, k: int) -> None:
    """Fill `selection[n_selected:k]` in place with the largest-marginal items not yet selected.

    `systematic_sample` calls this as its float-drift guard: floating-point accumulation in its
    main loop can drift a grid point across an interval edge, ending the traversal with fewer
    than `k` selected.
    """
    n = marginals.shape[0]
    selected_mask = np.zeros(n, dtype=np.bool_)
    for t in range(n_selected):
        selected_mask[selection[t]] = True
    while n_selected < k:
        best = np.int64(-1)
        best_p = -1.0
        for j in range(n):
            if not selected_mask[j] and marginals[j] > best_p:
                best_p = marginals[j]
                best = j
        if best < 0:
            # non-finite marginals (a bug upstream) compare False everywhere; degrade to the
            # first unselected item so the guard always yields a VALID selection, never a -1
            # written into it
            for j in range(n):
                if not selected_mask[j]:
                    best = j
                    break
        selected_mask[best] = True
        selection[n_selected] = best
        n_selected += 1


@numba.njit(cache=True)
def sample_and_repair(
    marginals: NDArray[np.float64],
    k: int,
    rng_state: NDArray[np.uint64],
    con_indices: NDArray[np.int32],
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
    max_swaps: int,
) -> tuple[NDArray[np.int64], float]:
    """Draw one systematic sample from the marginals and greedily swap-repair it.

    Returns:
        `(selection, violation)`: the repaired selection of `k` item indices and its weighted
        violation (0 means the draw ended feasible — a witness).

    Args:
        marginals: per-item inclusion probabilities in [0, 1], summing to `k`.
        k: the selection size.
        rng_state: xoroshiro128+ state, advanced in place.
        con_indices: packed constraint->item membership array (`ConstraintList.to_numpy`).
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
        max_swaps: the repair swap budget.
    """
    n = marginals.shape[0]
    m = con_min.shape[0]
    selection = systematic_sample(marginals, k, rng_state)
    sel_mask = np.zeros(n, dtype=np.bool_)
    for t in range(k):
        sel_mask[selection[t]] = True
    counts = _selection_counts(item_indptr, item_cons, selection, m)
    violation = _repair_selection(
        con_indices, item_indptr, item_cons, con_min, con_max, weights, sel_mask, counts, max_swaps
    )
    out = np.empty(k, dtype=np.int64)
    t = 0
    for j in range(n):
        if sel_mask[j]:
            out[t] = j
            t += 1
    return out, violation


@numba.njit(cache=True)
def deterministic_round(
    marginals: NDArray[np.float64],
    k: int,
    con_indices: NDArray[np.int32],
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
    max_swaps: int,
) -> tuple[NDArray[np.int64], float]:
    """Take the `k` largest marginals as the selection and greedily swap-repair it.

    This is the seed-free baseline beside the randomized draws: cheap, but it discards the
    spread the marginals encode.

    Returns:
        `(selection, violation)` as in `sample_and_repair`.

    Args:
        marginals: per-item inclusion probabilities in [0, 1], summing to `k`.
        k: the selection size.
        con_indices: packed constraint->item membership array (`ConstraintList.to_numpy`).
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
        max_swaps: the repair swap budget.
    """
    n = marginals.shape[0]
    m = con_min.shape[0]
    selection = np.argsort(marginals)[-k:]
    sel_mask = np.zeros(n, dtype=np.bool_)
    for t in range(k):
        sel_mask[selection[t]] = True
    counts = _selection_counts(item_indptr, item_cons, selection, m)
    violation = _repair_selection(
        con_indices, item_indptr, item_cons, con_min, con_max, weights, sel_mask, counts, max_swaps
    )
    out = np.empty(k, dtype=np.int64)
    t = 0
    for j in range(n):
        if sel_mask[j]:
            out[t] = j
            t += 1
    return out, violation


@numba.njit(cache=True)
def selection_violation(
    selection: NDArray[np.int64],
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> float:
    """Return the weighted violation of a selection (0 means it satisfies every constraint).

    Args:
        selection: the selected item indices.
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
    """
    counts = _selection_counts(item_indptr, item_cons, selection, con_min.shape[0])
    return _weighted_violation(counts, con_min, con_max, weights)
