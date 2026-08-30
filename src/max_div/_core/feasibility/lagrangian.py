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

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core._random import new_rng_state, rand_nz_float64

from .evaluation import (
    G_POSITIVE_TOL,
    _counts_feasible,
    _dual_value,
    _item_scores,
    _per_constraint_violation,
    _selection_counts,
    _top_k_items,
)
from .indexing import _constraint_set_sizes, build_item_constraint_csr
from .repair import SWAP_CAP_MIN, SWAP_CAP_PER_K, _repair_selection
from .result import FeasibilityResult, FeasibilityStatus

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

# candidate generation
N_ROUNDS = 16  # candidate selections tried before giving up
N_NOISE_FREE_ROUNDS = 2  # pure top-k candidates (alternating multiplier sets) before noise starts
NOISE_SCALE = 0.05  # score-noise magnitude, as a fraction of the mean constraint weight


# dual-ascent iteration-budget grades — callers pick a grade rather than passing a raw count
FEASIBILITY_MAX_ITER_LOW = 200  # low value when speed is prioritized
FEASIBILITY_MAX_ITER_MEDIUM = 2000  # reasonable default value
FEASIBILITY_MAX_ITER_HIGH = 8000  # high value when result quality is prioritized


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
) -> tuple[float, NDArray[np.int64]]:
    """Generate candidate selections from the mature scores and swap-repair each.

    Rounds alternate the two score vectors; the un-noised top-k of each is the single best guess,
    so noise starts only after both have had their pure round.  The loop stops early once the best
    violation reaches `stop_at_floor` — pass 0.0 to stop at a witness, or the certified floor to
    stop at a provably optimal least-infeasible selection.

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

    Returns:
        `(best_violation, best_selection)` over all rounds.
    """
    m = con_min.shape[0]
    noise_scale = NOISE_SCALE * (np.sum(weights) / m) if m > 0 else 0.0
    max_swaps = max(SWAP_CAP_MIN, SWAP_CAP_PER_K * k)
    rng_state = new_rng_state(np.int64(seed))

    best_violation = np.inf
    best_selection = np.empty(0, dtype=np.int64)
    for r in range(N_ROUNDS):
        scores = scores_avg.copy() if r % 2 == 0 else scores_best.copy()
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
) -> tuple[int, NDArray[np.int64], float, NDArray[np.int64], float, NDArray[np.float64], NDArray[np.float64]]:
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
        no_violation = np.zeros(con_min.shape[0], dtype=np.int64)
        return _FEASIBLE, witness, 0.0, no_violation, best_g, lam_min_best, lam_max_best

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
    )

    # phase 4 — assemble the verdict: proofs win over construction outcomes; a zero-violation
    # construction is a witness; anything else is UNKNOWN
    m = con_min.shape[0]
    per_constraint = _per_constraint_violation(
        _selection_counts(item_indptr, item_cons, best_selection, m), con_min, con_max
    )
    if certified_infeasible:
        return _INFEASIBLE, best_selection, best_violation, per_constraint, best_g, lam_min_best, lam_max_best
    if best_violation <= 0.0:
        return _FEASIBLE, best_selection, 0.0, per_constraint, best_g, lam_min_best, lam_max_best
    return _UNKNOWN, best_selection, best_violation, per_constraint, best_g, lam_min_best, lam_max_best


def find_feasible(
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    con_weights: NDArray[np.float64],
    n: int,
    k: int,
    max_iter: int,
    seed: int = 0,
    stop_at_first_proof: bool = False,
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
        max_iter: ascent iteration budget; cost is proportional, and a larger budget can only
            move an unknown verdict toward a proof.
        seed: seed for the candidate-generation noise (the ascent itself is deterministic).
        stop_at_first_proof: exit the ascent as soon as infeasibility is proven, forgoing the
            mature bound and the scores candidate generation draws from (verdict mode).

    Returns:
        A `FeasibilityResult` (see its docstring for the field semantics).  The certificate it may
        carry is independently verifiable: recompute the scores from `(lam_min, lam_max)`, sum the
        k largest, and check `lam_min @ min_counts - lam_max @ max_counts - topk_sum` is positive.
    """
    con_min = con_values[:, 0].astype(np.int64)
    con_max = con_values[:, 1].astype(np.int64)
    status, selection, violation, per_constraint, bound, lam_min, lam_max = _find_feasible(
        con_indices,
        con_min,
        con_max,
        con_weights.astype(np.float64),
        int(n),
        int(k),
        int(max_iter),
        int(seed),
        bool(stop_at_first_proof),
    )
    return FeasibilityResult(
        status=FeasibilityStatus(int(status)),
        selection=selection,
        violation=violation,
        violation_per_constraint=per_constraint,
        bound=bound,
        lam_min=lam_min,
        lam_max=lam_max,
    )
