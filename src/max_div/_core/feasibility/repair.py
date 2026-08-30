"""Greedy swap repair: local search that drives a selection's weighted violation down in place.

Each round targets the worst-violated constraint, pairs the best primary move among its members
with the best size-preserving counterpart, and executes only strictly improving pairs.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from .evaluation import _weighted_violation


SWAP_CAP_PER_K = 10  # repair swap budget per unit of k ...
SWAP_CAP_FLOOR = 500  # ... floored here (strict improvement already guarantees termination)
TIE_TOL = 1e-12  # improvement threshold for repair swaps


@numba.njit(cache=True)
def _add_gain(
    item: int,
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    counts: NDArray[np.int64],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> float:
    """Return the estimated violation decrease of adding `item` at the current counts.

    Each constraint containing the item contributes `+w` when short (the addition helps) and `-w`
    when at or above its max (the addition creates or worsens excess).  An estimate, not the exact
    delta: it ignores interactions when the paired removal shares constraints with this item.

    Args:
        item: the item whose addition is evaluated.
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        counts: per-constraint counts of selected members.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
    """
    gain = 0.0
    for e in range(item_indptr[item], item_indptr[item + 1]):
        i = item_cons[e]
        if counts[i] < con_min[i]:
            gain += weights[i]
        elif counts[i] >= con_max[i]:
            gain -= weights[i]
    return gain


@numba.njit(cache=True)
def _remove_gain(
    item: int,
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    counts: NDArray[np.int64],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> float:
    """Return the estimated violation decrease of removing `item` at the current counts.

    Mirror image of `_add_gain`: `+w` when the constraint is in excess, `-w` when the removal
    would create or worsen a shortfall.

    Args:
        item: the item whose removal is evaluated.
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        counts: per-constraint counts of selected members.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
    """
    gain = 0.0
    for e in range(item_indptr[item], item_indptr[item + 1]):
        i = item_cons[e]
        if counts[i] > con_max[i]:
            gain += weights[i]
        elif counts[i] <= con_min[i]:
            gain -= weights[i]
    return gain


@numba.njit(cache=True)
def _worst_violated_constraint(
    counts: NDArray[np.int64],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> int:
    """Return the index of the constraint with the largest weighted violation, or -1 if none.

    Args:
        counts: per-constraint counts of selected members.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
    """
    worst = -1
    worst_v = 0.0
    for i in range(counts.shape[0]):
        if counts[i] < con_min[i]:
            v = weights[i] * (con_min[i] - counts[i])
        elif counts[i] > con_max[i]:
            v = weights[i] * (counts[i] - con_max[i])
        else:
            v = 0.0
        if v > worst_v:
            worst_v = v
            worst = i
    return worst


@numba.njit(cache=True)
def _best_member_move(
    worst: int,
    short: bool,
    con_indices: NDArray[np.int32],
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    sel_mask: NDArray[np.bool_],
    counts: NDArray[np.int64],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> tuple[int, float]:
    """Return the best primary move inside the worst constraint's member set.

    When the constraint is `short`, that is the unselected member whose addition gains most;
    otherwise the selected member whose removal gains most.  Returns `(item, gain)`, item -1 when
    no candidate exists.

    Args:
        worst: the constraint whose violation the move targets.
        short: whether that constraint is below its minimum (else it is in excess).
        con_indices: packed constraint->item membership array (`ConstraintList.to_numpy`).
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        sel_mask: boolean selection mask over items.
        counts: per-constraint counts of selected members.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
    """
    best_item = -1
    best_gain = 0.0
    for e in range(con_indices[2 * worst], con_indices[2 * worst + 1]):
        j = con_indices[e]
        if sel_mask[j] == short:  # short needs an unselected member; excess a selected one
            continue
        if short:
            gain = _add_gain(j, item_indptr, item_cons, counts, con_min, con_max, weights)
        else:
            gain = _remove_gain(j, item_indptr, item_cons, counts, con_min, con_max, weights)
        if gain > best_gain:
            best_gain = gain
            best_item = j
    return best_item, best_gain


@numba.njit(cache=True)
def _best_counterpart_move(
    primary_item: int,
    short: bool,
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    sel_mask: NDArray[np.bool_],
    counts: NDArray[np.int64],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> tuple[int, float]:
    """Return the best counterpart move keeping the selection size at k.

    A `short` primary move adds a member, so the counterpart removes the selected item whose
    removal gains most (usually a negative gain — the removal that loses least); an excess primary
    move symmetrically adds the best unselected item.  Returns `(item, gain)`, item -1 when none exists.

    Args:
        primary_item: the item of the primary move, excluded from counterpart candidates.
        short: whether the primary move was an addition (else a removal).
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        sel_mask: boolean selection mask over items.
        counts: per-constraint counts of selected members.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
    """
    n = item_indptr.shape[0] - 1
    best_item = -1
    best_gain = -np.inf
    for j in range(n):
        if sel_mask[j] != short or j == primary_item:
            continue
        if short:
            gain = _remove_gain(j, item_indptr, item_cons, counts, con_min, con_max, weights)
        else:
            gain = _add_gain(j, item_indptr, item_cons, counts, con_min, con_max, weights)
        if gain > best_gain:
            best_gain = gain
            best_item = j
    return best_item, best_gain


@numba.njit(cache=True)
def _apply_swap(
    add_item: int,
    remove_item: int,
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    sel_mask: NDArray[np.bool_],
    counts: NDArray[np.int64],
) -> None:
    """Execute one add/remove pair, updating the selection mask and counts in place.

    Args:
        add_item: the item entering the selection.
        remove_item: the item leaving the selection.
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        sel_mask: boolean selection mask over items, updated in place.
        counts: per-constraint counts of selected members, updated in place.
    """
    sel_mask[add_item] = True
    sel_mask[remove_item] = False
    for e in range(item_indptr[add_item], item_indptr[add_item + 1]):
        counts[item_cons[e]] += 1
    for e in range(item_indptr[remove_item], item_indptr[remove_item + 1]):
        counts[item_cons[e]] -= 1


@numba.njit(cache=True)
def _repair_selection(
    con_indices: NDArray[np.int32],
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
    sel_mask: NDArray[np.bool_],
    counts: NDArray[np.int64],
    max_swaps: int,
) -> float:
    """Greedily swap-repair a selection in place; return its final weighted violation.

    Each round targets the worst-violated constraint, pairs the best primary move among its
    members with the best size-preserving counterpart, and executes only strictly improving pairs.
    The loop stops at violation zero, at a stall (no candidate or no strict improvement), or at
    the swap cap.  Single swaps cannot perform the coordinated multi-swap exchanges some feasible
    structures require — those land in the UNKNOWN outcome by design.

    Args:
        con_indices: packed constraint->item membership array (`ConstraintList.to_numpy`).
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
        sel_mask: boolean selection mask over items, repaired in place.
        counts: per-constraint counts of selected members, updated in place.
        max_swaps: the swap budget (safety cap; strict improvement terminates on its own for
            integral weights).
    """
    for _ in range(max_swaps):
        worst = _worst_violated_constraint(counts, con_min, con_max, weights)
        if worst == -1:
            return 0.0
        short = counts[worst] < con_min[worst]
        primary, primary_gain = _best_member_move(
            worst, short, con_indices, item_indptr, item_cons, sel_mask, counts, con_min, con_max, weights
        )
        if primary == -1:
            return _weighted_violation(counts, con_min, con_max, weights)
        counterpart, counterpart_gain = _best_counterpart_move(
            primary, short, item_indptr, item_cons, sel_mask, counts, con_min, con_max, weights
        )
        if counterpart == -1 or primary_gain + counterpart_gain <= TIE_TOL:
            return _weighted_violation(counts, con_min, con_max, weights)
        if short:
            _apply_swap(primary, counterpart, item_indptr, item_cons, sel_mask, counts)
        else:
            _apply_swap(counterpart, primary, item_indptr, item_cons, sel_mask, counts)
    return _weighted_violation(counts, con_min, con_max, weights)

