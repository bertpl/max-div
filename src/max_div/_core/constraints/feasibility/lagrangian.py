"""The Lagrangian-relaxation pipeline decides whether a feasible selection exists and, where possible, constructs one.

The question it answers: can exactly `k` of `n` items be selected such that every constraint `i`
counts between `min_count_i` and `max_count_i` selected members?  Constraint sets may overlap
arbitrarily, which makes the decision NP-complete, so the pipeline returns one of three statuses:

- `FEASIBLE`: a witness selection satisfying every constraint is returned.
- `INFEASIBLE`: multipliers `(lam_min, lam_max)` with a positive dual value are returned — a proof that no
  feasible selection exists, and a lower bound on the weighted violation of every possible
  selection (`find_feasible` documents the verification).
- `UNKNOWN`: neither was found.  This verdict carries no information; callers must behave as if
  nothing was learned.

The mechanism works in three phases:

- Per-constraint violation prices `(lam_min_i, lam_max_i)` turn the problem into per-item scores, whose
  exact top-k yields the dual value `g`; `g > 0` proves infeasibility (`_dual_value`).
- Projected supergradient ascent searches for prices with high `g` (`_dual_ascent`); prices at the
  end of a full ascent are called "mature" throughout this module.
- The ascent returns a witness immediately when an iterate's top-k happens to satisfy every
  constraint; otherwise — since `g <= 0` proves nothing — witnesses are constructed separately:
  candidate top-k selections drawn from the mature scores, each finished by greedy swap repair.

Violation is weighted-linearly throughout: `sum of w_i * (shortfall_i + excess_i)`, with `w_i` the
user-set constraint weights.  This is a deliberate best effort at honoring relative constraint
importance — a quadratic penalty cannot be mapped onto this machinery, because the collapse of the
priced penalty into per-item scores requires the penalty to be linear in the counts.  The
consequence splits cleanly: verdicts and witnesses are penalty-shape-independent (zero violation
is zero violation under any shape), while the certified floor and the least-infeasible grading
are specific to the weighted-linear aggregate.

Soundness invariant: the ascent must evaluate `g` through an EXACT, unperturbed top-k — noise
there can overestimate `g` and fabricate a false infeasibility proof.  All randomization lives
exclusively in candidate generation, where no bound is claimed.

All array-level functions are numba-compiled; the pipeline runs once per call (not per solver
iteration), so clarity wins over micro-optimization everywhere outside the inner passes.
Docstrings here are deliberately fuller than elsewhere in the code base: correctness rests on
mathematical facts the code cannot show (why a top-k minimizes the priced penalty, why a positive
`g` is a proof), so each function states the fact it relies on.
"""

import math
from dataclasses import dataclass
from enum import IntEnum

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core._math import select_k_max
from max_div._core._random import new_rng_state, rand_nz_float64


# =================================================================================================
#  FeasibilityStatus / FeasibilityResult
# =================================================================================================
class FeasibilityStatus(IntEnum):
    """Three-valued outcome of the feasibility pipeline; the definite verdicts are proofs."""

    INFEASIBLE = -1
    UNKNOWN = 0
    FEASIBLE = 1


@dataclass(frozen=True)
class FeasibilityResult:
    """Outcome of `find_feasible`: the status plus the objects backing it.

    Attributes:
        status: `FEASIBLE` (selection is a witness), `INFEASIBLE` (bound and multipliers form a
            verifiable proof; selection is the best least-infeasible found), or `UNKNOWN` (no
            claim; selection is the least-violating found).
        selection: the constructed selection of k item indices.
        bound: the best dual value — when positive, a certified lower bound on the weighted
            violation of every possible selection, re-checkable from the multipliers alone.
        violation: the weighted violation of `selection` (0 for a witness).
        lam_min: the shortfall multipliers behind `bound`; with `lam_max` they form the
            infeasibility certificate when `status` is `INFEASIBLE`.
        lam_max: the excess multipliers behind `bound`.
    """

    status: FeasibilityStatus
    selection: NDArray[np.int64]
    bound: float
    violation: float
    lam_min: NDArray[np.float64]
    lam_max: NDArray[np.float64]


# =================================================================================================
#  Constants
# =================================================================================================
# njit-internal integer aliases of `FeasibilityStatus` — the compiled core works with plain ints,
# and the `find_feasible` wrapper converts back to the enum
_FEASIBLE = int(FeasibilityStatus.FEASIBLE)
_INFEASIBLE = int(FeasibilityStatus.INFEASIBLE)
_UNKNOWN = int(FeasibilityStatus.UNKNOWN)

# ascent
GAMMA0 = 1.0  # initial step-size scale of the projected supergradient ascent
STALL_WINDOW = 100  # iterations without a new best g before the step size is damped
STALL_SHRINK = 0.7  # damping factor applied to the step-size scale on a stall
VERDICT_MAX_ITER = 2000  # ascent budget when only a fast verdict is wanted

# candidate generation
N_ROUNDS = 16  # candidate selections tried before giving up
N_NOISE_FREE_ROUNDS = 2  # pure top-k candidates (alternating multiplier sets) before noise starts
NOISE_SCALE = 0.05  # score-noise magnitude, as a fraction of the mean constraint weight
PRIOR_EPS = 1e-12  # added to the normalized diversity prior so a zero-prior item stays finite

# repair
SWAP_CAP_PER_K = 10  # repair swap budget per unit of k ...
SWAP_CAP_FLOOR = 500  # ... floored here (strict improvement already guarantees termination)

# soundness margins
G_POSITIVE_TOL = 1e-9  # g must exceed this before it counts as an infeasibility proof
TIE_TOL = 1e-12  # improvement threshold for repair swaps

# construction-mode iteration budget
BUDGET_FRACTION = 0.10  # share of the solve budget granted to the ascent
EST_SEC_PER_OP = 2e-9  # nominal cost of one inner-pass operation, for the time-typed budget
CONSTRUCTION_MIN_ITER = 500
CONSTRUCTION_MAX_ITER = 8000
CONSTRUCTION_DEFAULT_ITER = 2000  # ascent budget for callers with no solve budget to derive one from


# =================================================================================================
#  Item <-> constraint indexing
# =================================================================================================
@numba.njit(cache=True)
def build_item_constraint_csr(con_indices: NDArray[np.int32], n: int) -> tuple[NDArray[np.int64], NDArray[np.int32]]:
    """Transpose the packed constraint->item layout into an item->constraint CSR (compressed sparse row) index.

    Args:
        con_indices: the packed representation built by `ConstraintList.to_numpy` — a 2m-element
            header of per-constraint [start, end) offsets, followed by the concatenated member
            indices.
        n: the number of items.

    Returns:
        `(item_indptr, item_cons)`: for item `j`, `item_cons[item_indptr[j]:item_indptr[j + 1]]`
        lists the constraints containing `j`.
    """
    item_indptr = np.zeros(n + 1, dtype=np.int64)
    if con_indices.shape[0] == 0:
        return item_indptr, np.empty(0, dtype=np.int32)
    m = con_indices[0] // 2

    # count memberships per item, then prefix-sum into indptr
    for e in range(2 * m, con_indices.shape[0]):
        item_indptr[con_indices[e] + 1] += 1
    for j in range(n):
        item_indptr[j + 1] += item_indptr[j]

    # fill, walking the constraints in order
    item_cons = np.empty(con_indices.shape[0] - 2 * m, dtype=np.int32)
    cursor = item_indptr[:-1].copy()
    for i in range(m):
        for e in range(con_indices[2 * i], con_indices[2 * i + 1]):
            j = con_indices[e]
            item_cons[cursor[j]] = np.int32(i)
            cursor[j] += 1
    return item_indptr, item_cons


@numba.njit(cache=True)
def _constraint_set_sizes(item_cons: NDArray[np.int32], m: int) -> NDArray[np.int64]:
    """Return the member count of each constraint.

    Args:
        item_cons: item->constraint CSR values — the constraints containing each item.
        m: the number of constraints.
    """
    sizes = np.zeros(m, dtype=np.int64)
    for e in range(item_cons.shape[0]):
        sizes[item_cons[e]] += 1
    return sizes


# =================================================================================================
#  Inner passes: scores, top-k, counts, violation, dual value
# =================================================================================================
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


# =================================================================================================
#  Dual ascent
# =================================================================================================
@numba.njit(cache=True)
def _binding_upper_mask(
    item_cons: NDArray[np.int32],
    con_max: NDArray[np.int64],
    m: int,
    k: int,
) -> NDArray[np.bool_]:
    """Return which upper bounds can actually bind: `max_count < min(k, set size)`.

    A constraint whose max count can never be exceeded needs no price; its `lam_max` stays 0 and its
    supergradient component is skipped.

    Args:
        item_cons: item->constraint CSR values — the constraints containing each item.
        con_max: per-constraint maximum counts.
        m: the number of constraints.
        k: the selection size.
    """
    sizes = _constraint_set_sizes(item_cons, m)
    active = np.empty(m, dtype=np.bool_)
    for i in range(m):
        active[i] = con_max[i] < min(k, sizes[i])
    return active


@numba.njit(cache=True)
def _projected_step(
    lam_min: NDArray[np.float64],
    lam_max: NDArray[np.float64],
    lam_max_active: NDArray[np.bool_],
    counts: NDArray[np.int64],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
    gamma: float,
    t: int,
) -> None:
    """Take one projected supergradient step in place.

    The supergradient at the current top-k is `(con_min - counts)` for `lam_min` and
    `(counts - con_max)` for `lam_max`: prices of starved constraints rise, prices of over-satisfied
    ones decay.  The step is `gamma / (sqrt(t) * max(gradient norm, 1))`, and each multiplier is
    projected back into its box `[0, weight_i]` — the cap that makes the matured dual value a
    quantitative violation floor rather than only a sign test.

    Args:
        lam_min: shortfall prices, updated in place.
        lam_max: excess prices, updated in place.
        lam_max_active: which upper bounds can bind (`_binding_upper_mask`); inactive entries stay 0.
        counts: per-constraint counts of the current top-k selection.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights — also the multiplier box bounds.
        gamma: the current step-size scale.
        t: the 1-based iteration number (drives the `1/sqrt(t)` decay).
    """
    m = con_min.shape[0]
    norm_sq = 0.0
    for i in range(m):
        g_min = float(con_min[i] - counts[i])
        norm_sq += g_min * g_min
        if lam_max_active[i]:
            g_max = float(counts[i] - con_max[i])
            norm_sq += g_max * g_max
    step = gamma / (np.sqrt(float(t)) * max(np.sqrt(norm_sq), 1.0))
    for i in range(m):
        lam_min[i] = min(max(lam_min[i] + step * (con_min[i] - counts[i]), 0.0), weights[i])
        if lam_max_active[i]:
            lam_max[i] = min(max(lam_max[i] + step * (counts[i] - con_max[i]), 0.0), weights[i])


@numba.njit(cache=True)
def _dual_ascent(
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
    k: int,
    max_iter: int,
    stop_at_first_proof: bool,
) -> tuple[
    float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]
]:
    """Run the projected supergradient ascent from zero multipliers.

    Each iteration evaluates the dual value `g` at the current prices through an exact top-k (the
    soundness invariant), tracks the best `g` seen, and steps the prices.  Two cheap early exits:

    - the iteration's top-k happens to satisfy every constraint -> that selection is a witness and
      the ascent returns it immediately (any feasible selection proves feasibility);
    - `stop_at_first_proof` and `g` exceeds the positivity margin -> infeasibility is proven and a
      fast verdict needs no matured bound.

    Without `stop_at_first_proof` the ascent runs to `max_iter` even after `g` goes positive: the
    mature bound is the certified violation floor, and the mature prices are the score vectors
    candidate generation draws from.

    Args:
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights — also the multiplier box bounds.
        k: the selection size.
        max_iter: the ascent iteration budget.
        stop_at_first_proof: exit as soon as `g` clears the positivity margin.

    Returns:
        `(best_g, lam_min_avg, lam_max_avg, lam_min_best, lam_max_best, witness)`.  The two
        multiplier pairs are complementary guesses for candidate generation: the `_best` pair is
        the prices at the highest `g` seen (the sharpest single guess), the `_avg` pair is the
        running average of all iterates, which smooths the zigzag of subgradient ascent and
        typically ranks near-tied items differently.  `witness` is the feasible selection from the
        early exit, empty when none was found.
    """
    m = con_min.shape[0]
    lam_min = np.zeros(m, dtype=np.float64)
    lam_max = np.zeros(m, dtype=np.float64)
    lam_min_avg = np.zeros(m, dtype=np.float64)
    lam_max_avg = np.zeros(m, dtype=np.float64)
    lam_min_best = np.zeros(m, dtype=np.float64)
    lam_max_best = np.zeros(m, dtype=np.float64)
    lam_max_active = _binding_upper_mask(item_cons, con_max, m, k)

    best_g = -np.inf
    gamma = GAMMA0
    stall = 0
    for t in range(1, max_iter + 1):
        scores = _item_scores(item_indptr, item_cons, lam_min, lam_max)
        selection = _top_k_items(scores, k)  # exact top-k: the soundness invariant
        counts = _selection_counts(item_indptr, item_cons, selection, m)

        if _counts_feasible(counts, con_min, con_max):
            return best_g, lam_min_avg, lam_max_avg, lam_min_best, lam_max_best, selection

        g = _dual_value(con_min, con_max, lam_min, lam_max, scores, selection)
        if g > best_g:
            best_g = g
            lam_min_best[:] = lam_min
            lam_max_best[:] = lam_max
            stall = 0
        else:
            stall += 1
            if stall >= STALL_WINDOW:
                gamma *= STALL_SHRINK
                stall = 0
        if stop_at_first_proof and best_g > G_POSITIVE_TOL:
            break

        _projected_step(lam_min, lam_max, lam_max_active, counts, con_min, con_max, weights, gamma, t)

        # running average of the iterates; averages smooth the zigzag of subgradient ascent and
        # give candidate generation a second, complementary score vector
        a = 1.0 / float(t)
        for i in range(m):
            lam_min_avg[i] += a * (lam_min[i] - lam_min_avg[i])
            lam_max_avg[i] += a * (lam_max[i] - lam_max_avg[i])

    return best_g, lam_min_avg, lam_max_avg, lam_min_best, lam_max_best, np.empty(0, dtype=np.int64)


# =================================================================================================
#  Candidate generation + swap repair
# =================================================================================================
@numba.njit(cache=True)
def _gumbel_noise(n: int, rng_state: NDArray[np.uint64]) -> NDArray[np.float64]:
    """Return n iid Gumbel(0, 1) samples (the classical randomized-rounding perturbation).

    Args:
        n: the number of samples.
        rng_state: xoroshiro128+ state, advanced in place.
    """
    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        u = rand_nz_float64(rng_state)
        out[j] = -np.log(-np.log(u))
    return out


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


# =================================================================================================
#  Pipeline
# =================================================================================================
@numba.njit(cache=True)
def _violation_floor(weights: NDArray[np.float64], best_g: float, certified_infeasible: bool) -> float:
    """Return the certified violation floor, rounded up to the next integer for integral weights.

    Integral weights make every achievable violation an integer, which sharpens the fractional
    dual bound.

    Args:
        weights: per-constraint violation weights.
        best_g: the mature dual bound.
        certified_infeasible: whether `best_g` cleared the positivity margin (only then is
            rounding up justified).
    """
    for i in range(weights.shape[0]):
        if abs(weights[i] - np.round(weights[i])) > 1e-12:
            return best_g
    return np.ceil(best_g - G_POSITIVE_TOL) if certified_infeasible else best_g


@numba.njit(cache=True)
def _candidate_rounds(
    con_indices: NDArray[np.int32],
    item_indptr: NDArray[np.int64],
    item_cons: NDArray[np.int32],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
    n: int,
    k: int,
    scores_avg: NDArray[np.float64],
    scores_best: NDArray[np.float64],
    seed: int,
    stop_at_floor: float,
    log_prior: NDArray[np.float64],
    beta: float,
) -> tuple[float, NDArray[np.int64]]:
    """Generate candidate selections from the mature scores and swap-repair each.

    Rounds alternate the two score vectors; the un-noised top-k of each is the single best guess,
    so noise starts only after both have had their pure round.  The loop stops early once the best
    violation reaches `stop_at_floor` — pass 0.0 to stop at a witness, or the certified floor to
    stop at a provably optimal least-infeasible selection.

    Candidate generation is the only place the `beta` tilt may be applied; this module's soundness
    invariant forbids perturbing the ascent's top-k.

    Args:
        con_indices: packed constraint->item membership array (`ConstraintList.to_numpy`).
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
        n: the number of items.
        k: the selection size.
        scores_avg: item scores at the running-average mature prices — the zigzag-smoothed guess.
        scores_best: item scores at the highest-g mature prices — the sharpest guess.
        seed: seed for the score noise.
        stop_at_floor: the violation level at which searching further cannot improve the answer.
        log_prior: per-item log diversity prior, or an empty array when no prior is supplied.
        beta: weight of `log_prior` in the candidate scores; 0 disables the tilt.

    Returns:
        `(best_violation, best_selection)` over all rounds.
    """
    m = con_min.shape[0]
    noise_scale = NOISE_SCALE * (np.sum(weights) / m) if m > 0 else 0.0
    max_swaps = max(SWAP_CAP_FLOOR, SWAP_CAP_PER_K * k)
    rng_state = new_rng_state(np.int64(seed))
    use_prior = beta != 0.0 and log_prior.shape[0] == n

    best_violation = np.inf
    best_selection = np.empty(0, dtype=np.int64)
    for r in range(N_ROUNDS):
        scores = scores_avg.copy() if r % 2 == 0 else scores_best.copy()
        if use_prior:
            for j in range(n):
                scores[j] += beta * log_prior[j]
        if r >= N_NOISE_FREE_ROUNDS:
            noise = _gumbel_noise(n, rng_state)
            for j in range(n):
                scores[j] += noise_scale * noise[j]
        selection = _top_k_items(scores, k)
        sel_mask = np.zeros(n, dtype=np.bool_)
        for t in range(k):
            sel_mask[selection[t]] = True
        counts = _selection_counts(item_indptr, item_cons, selection, m)
        violation = _repair_selection(
            con_indices, item_indptr, item_cons, con_min, con_max, weights, sel_mask, counts, max_swaps
        )
        if violation < best_violation:
            best_violation = violation
            best_selection = np.where(sel_mask)[0].astype(np.int64)
            if best_violation <= stop_at_floor + G_POSITIVE_TOL:
                break
    return best_violation, best_selection


@numba.njit(cache=True)
def _find_feasible(
    con_indices: NDArray[np.int32],
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    weights: NDArray[np.float64],
    n: int,
    k: int,
    max_iter: int,
    seed: int,
    stop_at_first_proof: bool,
    log_prior: NDArray[np.float64],
    beta: float,
) -> tuple[int, NDArray[np.int64], float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Run the numba core of the pipeline: ascent, then candidate construction, then the verdict.

    Args:
        con_indices: packed constraint->item membership array (`ConstraintList.to_numpy`).
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        weights: per-constraint violation weights.
        n: the number of items.
        k: the selection size.
        max_iter: the ascent iteration budget.
        seed: seed for the candidate-generation noise (the ascent itself is deterministic).
        stop_at_first_proof: exit the ascent at the first infeasibility proof (verdict mode).
        log_prior: per-item log diversity prior for candidate generation, or an empty array.
        beta: weight of `log_prior` in the candidate scores; 0 disables the tilt.

    Returns:
        The integer-status tuple the `find_feasible` wrapper packs into a `FeasibilityResult`.
    """
    item_indptr, item_cons = build_item_constraint_csr(con_indices, n)

    # phase 1 — price ascent: matures the bound and the multipliers, and may already hand back a
    # witness (a mid-ascent top-k that satisfies every constraint)
    best_g, lam_min_avg, lam_max_avg, lam_min_best, lam_max_best, witness = _dual_ascent(
        item_indptr, item_cons, con_min, con_max, weights, k, max_iter, stop_at_first_proof
    )
    if witness.shape[0] > 0:
        return _FEASIBLE, witness, best_g, 0.0, lam_min_best, lam_max_best

    # phase 2 — interpret the bound: a positive g proves infeasibility, and its value becomes the
    # certified violation floor candidate construction can stop at
    certified_infeasible = best_g > G_POSITIVE_TOL
    floor = _violation_floor(weights, best_g, certified_infeasible)

    # phase 3 — construct: candidate top-k selections from both mature score vectors, each
    # finished by greedy swap repair
    scores_avg = _item_scores(item_indptr, item_cons, lam_min_avg, lam_max_avg)
    scores_best = _item_scores(item_indptr, item_cons, lam_min_best, lam_max_best)
    best_violation, best_selection = _candidate_rounds(
        con_indices,
        item_indptr,
        item_cons,
        con_min,
        con_max,
        weights,
        n,
        k,
        scores_avg,
        scores_best,
        seed,
        floor if certified_infeasible else 0.0,
        log_prior,
        beta,
    )

    # phase 4 — assemble the verdict: proofs win over construction outcomes; a zero-violation
    # construction is a witness; anything else is UNKNOWN
    if certified_infeasible:
        return _INFEASIBLE, best_selection, best_g, best_violation, lam_min_best, lam_max_best
    if best_violation <= 0.0:
        return _FEASIBLE, best_selection, best_g, 0.0, lam_min_best, lam_max_best
    return _UNKNOWN, best_selection, best_g, best_violation, lam_min_best, lam_max_best


def _log_diversity_prior(diversity_prior: NDArray[np.floating] | None) -> NDArray[np.float64]:
    """Normalize a diversity prior into the per-item log-probabilities candidate generation adds.

    Negative entries are clamped away, and `PRIOR_EPS` keeps a zero-prior item finite so it is
    disfavored rather than excluded.  A missing or all-zero prior carries no preference, so the
    empty array is returned and the tilt stays off.
    """
    if diversity_prior is None:
        return np.empty(0, dtype=np.float64)
    p = np.maximum(np.asarray(diversity_prior, dtype=np.float64), 0.0)
    total = p.sum()
    if total <= 0.0:
        return np.empty(0, dtype=np.float64)
    return np.log(p / total + PRIOR_EPS)


def find_feasible(
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    con_weights: NDArray[np.float64],
    n: int,
    k: int,
    max_iter: int,
    seed: int = 0,
    stop_at_first_proof: bool = False,
    diversity_prior: NDArray[np.floating] | None = None,
    beta: float = 0.0,
) -> FeasibilityResult:
    """Run the full feasibility pipeline: dual ascent, candidate generation, swap repair.

    Args:
        con_values: `(m, 2)` per-constraint `[min_count, max_count]`, as built by
            `ConstraintList.to_numpy` (the pristine problem-level values, not remaining counts).
        con_indices: packed constraint membership array from the same conversion.
        con_weights: per-constraint weights (strictly positive) — the user-set `Constraint.weight`
            values; the module docstring describes how they define the violation aggregate.
        n: number of items.
        k: selection size.
        max_iter: ascent iteration budget; see `construction_iteration_budget_seconds` /
            `construction_iteration_budget_iterations` for the budget-derived value, or
            `VERDICT_MAX_ITER` for a fast verdict.
        seed: seed for the candidate-generation noise (the ascent itself is deterministic).
        stop_at_first_proof: exit the ascent as soon as infeasibility is proven, forgoing the
            mature bound and the scores candidate generation draws from (verdict mode).
        diversity_prior: optional per-item non-negative preference, normalized here; consulted
            only when `beta` is nonzero.
        beta: how strongly candidate generation is tilted toward high-prior items, via a
            `beta * log(p)` term on the candidate scores; 0 disables the tilt.  A tilt cannot
            reach the ascent, and so changes which selection is constructed, never the verdict.

    Returns:
        A `FeasibilityResult` (see its docstring for the field semantics).  The certificate it may
        carry is independently verifiable: recompute the scores from `(lam_min, lam_max)`, sum the
        k largest, and check `lam_min @ min_counts - lam_max @ max_counts - topk_sum` is positive.
    """
    con_min = con_values[:, 0].astype(np.int64)
    con_max = con_values[:, 1].astype(np.int64)
    status, selection, bound, violation, lam_min, lam_max = _find_feasible(
        con_indices,
        con_min,
        con_max,
        con_weights.astype(np.float64),
        int(n),
        int(k),
        int(max_iter),
        int(seed),
        bool(stop_at_first_proof),
        _log_diversity_prior(diversity_prior),
        float(beta),
    )
    return FeasibilityResult(
        status=FeasibilityStatus(int(status)),
        selection=selection,
        bound=bound,
        violation=violation,
        lam_min=lam_min,
        lam_max=lam_max,
    )


# =================================================================================================
#  Construction-mode iteration budgets
# =================================================================================================
def construction_iteration_budget_seconds(t_max_sec: float, n: int, n_memberships: int) -> int:
    """Return the ascent iteration budget for a time-typed solve budget.

    The budget is a fixed share (`BUDGET_FRACTION`) of the solve budget, converted to iterations
    through a nominal per-iteration cost model — deterministic given problem and configuration, so
    same-seed reproducibility is preserved (machine-speed variation only shifts the actual wall
    fraction spent).

    Args:
        t_max_sec: the solve's total wall-clock budget in seconds.
        n: the number of items.
        n_memberships: the total constraint membership count (drives the per-iteration cost).
    """
    est_iter_cost_sec = (n_memberships + n * math.log2(max(n, 2))) * EST_SEC_PER_OP
    target = BUDGET_FRACTION * t_max_sec / est_iter_cost_sec
    return int(min(max(round(target), CONSTRUCTION_MIN_ITER), CONSTRUCTION_MAX_ITER))


def construction_iteration_budget_iterations(n_solver_iterations: int) -> int:
    """Return the ascent iteration budget for an iteration-typed solve budget.

    The same share is applied to the solver iteration count directly — no cost model involved, so
    the machine independence of iteration-typed budgets is preserved; the crudeness of equating
    ascent and solver iteration costs is absorbed by the clamps.

    Args:
        n_solver_iterations: the solve's total iteration budget.
    """
    target = BUDGET_FRACTION * n_solver_iterations
    return int(min(max(round(target), CONSTRUCTION_MIN_ITER), CONSTRUCTION_MAX_ITER))
