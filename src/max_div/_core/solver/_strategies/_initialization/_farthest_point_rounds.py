"""Farthest-point sampling in rounds draws many items per pass over the dataset.

A round collects the `candidate_pool_size` highest-contribution not-selected items into a pool with
one pass over the dataset, then draws from that pool without touching the dataset again, ending once
the pool can no longer be shown to hold the dataset's best candidates (see `_draw_round`).

Every draw ranges over the same candidates as picking one item at a time, so selections are of
equal quality but not identical, and rounds are several times faster at large n.

Rounds apply only where `are_farthest_point_rounds_supported` holds; `InitFarthestPoint` decides
when to use them.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core._random import randint
from max_div._core.metrics import DiversityContributionFamily, DiversityObjective
from max_div._core.metrics._distance import DISTANCE_STORE_TYPE, DistanceStore, get_distance
from max_div._core.solver._solver_state import SolverState


# =================================================================================================
#  Rounds
# =================================================================================================
def are_farthest_point_rounds_supported(objective: DiversityObjective) -> bool:
    """Return whether rounds apply: every term of `objective` is a separation-family metric over one distance.

    The rule that ends a round relies on each contribution only falling as items are selected, and
    only that kind of objective guarantees it.
    """
    specs = objective.distinct_tracker_specs
    return len(specs) == 1 and specs[0].contribution_family == DiversityContributionFamily.SEPARATION


def draw_farthest_point_round(
    state: SolverState, top_k: int, candidate_pool_size: int, k_remaining: int | np.int32, rng_state: NDArray[np.uint64]
) -> NDArray[np.int32]:
    """Draw one round's batch for a non-empty selection: collect a candidate pool and draw from it.

    The batch holds at least 1 item and grows while the pool can still be shown to hold the
    dataset's best candidates. Every draw samples uniformly among the `top_k` best remaining pool
    candidates and advances `rng_state` in place.
    """
    # --- candidate pool -------------------------
    # the candidate_pool_size highest-contribution not-selected items
    cand_idx, cand_val = state.top_not_selected_contributions(candidate_pool_size)
    # every item outside the pool is below the pool's lowest value, so a pool candidate still at
    # or above that value is among the dataset's best: it is the round's admission threshold
    threshold = np.float32(cand_val.min())
    if len(cand_idx) < candidate_pool_size:
        # the pool holds every remaining item, so nothing is outside it: draw until the pool is empty
        threshold = np.float32(-np.inf)

    # --- draw -----------------------------------
    b_target = min(len(cand_idx), int(k_remaining))
    out_batch = np.empty(b_target, dtype=np.int32)
    top_positions = np.empty(top_k, dtype=np.int32)  # scratch for the draw loop's top-k positions
    n_drawn = _draw_round(
        cand_idx,
        cand_val,
        np.int32(top_k),
        threshold,
        np.int64(b_target),
        state.distance_store,
        rng_state,
        out_batch,
        top_positions,
    )
    return out_batch[:n_drawn]


# =================================================================================================
#  Helpers
# =================================================================================================
@numba.njit(
    numba.void(numba.float32[:], numba.int32[:], numba.int64, numba.int64),
    inline="always",
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _sift_down(values: NDArray[np.float32], positions: NDArray[np.int32], size: np.int64, start: np.int64) -> None:
    """Sink the entry at `start` until every parent again holds a value no larger than its children's.

    `positions` is laid out as a binary heap (see `_select_highest` for the layout): the children
    of slot i sit at 2i+1 and 2i+2. A parent that has become larger than a child is swapped with its
    smaller child, and the check repeats one level down until the entry rests below no larger
    value or reaches the bottom.
    """
    i_parent = start
    while True:
        i_left = 2 * i_parent + 1
        i_right = i_left + 1
        i_min = i_parent
        if i_left < size and values[positions[i_left]] < values[positions[i_min]]:
            i_min = i_left
        if i_right < size and values[positions[i_right]] < values[positions[i_min]]:
            i_min = i_right
        if i_min == i_parent:
            return
        positions[i_parent], positions[i_min] = positions[i_min], positions[i_parent]
        i_parent = i_min


@numba.njit(
    numba.float32(numba.float32[:], numba.int64, numba.int64, numba.int32[:]),
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _select_highest(
    values: NDArray[np.float32], n_live: np.int64, n_top: np.int64, out_positions: NDArray[np.int32]
) -> np.float32:
    r"""Fill `out_positions` with the `n_top` highest values' positions among the first `n_live`; return the lowest.

    The positions come back in unspecified order. Ties keep the earlier position, so with
    `n_top == 1` the result is the first maximum. The caller owns `out_positions` (at least
    `n_top` long), so a draw allocates nothing.

    How it works: `out_positions` is kept as a min-heap of the `n_top` best candidates seen so far —
    a binary tree stored in an array, where slot i's children are slots 2i+1 and 2i+2 and every
    parent's value is no larger than its children's, so slot 0 always holds the *smallest* of the
    kept values:

              v0                 v0 <= v1, v2
            /    \\               v1 <= v3, v4
          v1      v2             v2 <= v5, v6     (siblings are not ordered among themselves)
         /  \\    /  \
        v3  v4  v5  v6

    The first `n_top` positions seed the heap, and each later value is compared with slot 0: a
    value no larger than the smallest kept value cannot belong to the top `n_top`, so it is skipped;
    a larger one replaces slot 0 and sinks to its place (`_sift_down`), evicting the old smallest.
    After the pass the heap holds exactly the `n_top` largest, and slot 0 — the smallest of them —
    is the `n_top`-th largest overall, which is the value returned. One pass, `log(n_top)` work per
    replacement, no sorting of the rest.
    """
    for i in range(n_top):
        out_positions[i] = i
    for i in range(n_top // 2 - 1, -1, -1):
        _sift_down(values, out_positions, n_top, np.int64(i))
    for i in range(n_top, n_live):
        if values[i] > values[out_positions[0]]:
            out_positions[0] = i
            _sift_down(values, out_positions, n_top, np.int64(0))
    return values[out_positions[0]]


@numba.njit(
    numba.int64(
        numba.int32[:],
        numba.float32[:],
        numba.int32,
        numba.float32,
        numba.int64,
        DISTANCE_STORE_TYPE,
        numba.uint64[:],
        numba.int32[:],
        numba.int32[:],
    ),
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _draw_round(
    cand_idx: NDArray[np.int32],
    cand_val: NDArray[np.float32],
    top_k: np.int32,
    threshold: np.float32,
    b_target: np.int64,
    store: DistanceStore,
    rng_state: NDArray[np.uint64],
    out_batch: NDArray[np.int32],
    top_positions: NDArray[np.int32],
) -> np.int64:
    """Draw items from the candidate pool into `out_batch` while it provably holds the dataset's best; return the count.

    `threshold` is the lowest contribution the pool held when the round opened. Every item outside
    the pool was below it then, and contributions only fall as items are selected, so a pool
    candidate still at or above it is among the whole dataset's best — which is what makes each
    draw range over the same candidates as picking one item at a time. The round ends when the
    pool's `top_k`-th best live candidate is below `threshold`, or when fewer than `top_k` live
    candidates remain while `threshold` is finite. After each draw the remaining pool is refreshed
    against the drawn item.

    `cand_idx` and `cand_val` are reordered and overwritten in place; `out_batch` needs `b_target`
    slots; `top_positions` is caller-owned scratch of at least `top_k` slots, so a draw allocates
    nothing.
    """
    count = len(cand_idx)
    p_uniform = np.zeros(0, dtype=np.float32)  # module globals freeze to readonly inside njit
    n_drawn = np.int64(0)
    for bi in range(b_target):
        n_live = np.int64(count - bi)
        k_eff = min(np.int64(top_k), n_live)
        if k_eff <= 0:
            break
        if k_eff < top_k and threshold > -np.inf:
            break  # too few live candidates to show the draw would range over the dataset's best
        if _select_highest(cand_val, n_live, k_eff, top_positions) < threshold:
            break
        # a single candidate needs no draw, matching `InitFarthestPoint`'s argmax pick for `top_k=1`
        slot = 0 if k_eff == 1 else randint(np.int32(k_eff), np.int32(1), False, p_uniform, rng_state)[0]
        pick_pos = top_positions[slot]
        x = cand_idx[pick_pos]
        out_batch[n_drawn] = x
        n_drawn += 1
        # remove the drawn candidate (swap in the last remaining one), then refresh the rest
        last = count - bi - 1
        cand_idx[pick_pos] = cand_idx[last]
        cand_val[pick_pos] = cand_val[last]
        for c in range(count - bi - 1):
            cand_val[c] = min(cand_val[c], get_distance(store, x, cand_idx[c]))
    return n_drawn
