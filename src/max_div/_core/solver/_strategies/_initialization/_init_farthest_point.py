import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core._math import select_k_max, select_k_max_into
from max_div._core._random import P_UNIFORM, randint
from max_div._core.metrics import DiversityContributionFamily, DiversityObjective
from max_div._core.metrics._distance import DISTANCE_STORE_TYPE, DistanceStore, get_distance
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


# =================================================================================================
#  InitFarthestPoint
# =================================================================================================
class InitFarthestPoint(InitializationStrategy):
    """Initialize by farthest-point sampling: a seeded random start item, then greedy picks.

    Each pick adds one of the `top_k` not-yet-selected items with the highest diversity contribution
    with respect to the current selection:

    - separation-family metrics: the items farthest from their nearest selected neighbor
      (classical farthest-point sampling);
    - `MEAN_PAIRWISE_DISTANCE`: the items with the highest mean distance to the selection
      (the greedy max-sum construction).

    How items are added depends on the solve's primary diversity objective:

    - **in rounds, from a limited candidate pool per round** (faster): each round collects the
      `candidate_pool_size` highest-contribution items into a pool with one pass over the dataset,
      then draws from that pool, refreshing the remaining candidates after each draw, until the pool
      can no longer be shown to hold the dataset's best candidates. Used when `candidate_pool_size`
      is not None and every term of the objective is a separation-family metric over one distance;
      several times faster at large n.
    - **one at a time, from all not-selected items** (slower): one pass over the dataset per pick.
      Used in every other case.

    Both offer each pick the same candidates, so selections are of equal quality but not identical.

    Constraints are ignored by design; feasibility is left to the optimization steps.

    Parameters:
    - top_k (int): every pick samples uniformly among the `top_k` highest contributions; `top_k=1`
                   is the exact argmax and consumes no randomness. (default: 8)
    - candidate_pool_size (int | None): how many candidates a round collects, which bounds how many
                                        items it can draw. A larger `candidate_pool_size` needs fewer
                                        passes over the dataset, but after each draw more pool
                                        candidates need their contribution updated; above a few
                                        hundred, those updates cost more than the saved passes.
                                        `candidate_pool_size` cannot affect the selection's quality,
                                        only the time spent. `None` picks one item at a time for
                                        every objective. (default: 256)

    Time Complexity:
       - ~O(n * k), times d when distances are computed on demand from vectors.
    """

    def __init__(self, top_k: int = 8, candidate_pool_size: int | None = 256) -> None:
        """Create the strategy.

        Raises:
            ValueError: If `top_k` is below 1, or `candidate_pool_size` is below `top_k`
                (a round could then not offer a full draw).
        """
        super().__init__()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if candidate_pool_size is not None and candidate_pool_size < top_k:
            raise ValueError(f"candidate_pool_size must be >= top_k ({top_k}), got {candidate_pool_size}")
        self._top_k = top_k
        self._candidate_pool_size = candidate_pool_size

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        if state.n_selected == 0:
            return randint(n=state.n, k=np.int32(1), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
        elif self._candidate_pool_size is not None and self._are_rounds_supported(state.primary_objective):
            return self._draw_round(state, self._candidate_pool_size, k_remaining)
        else:
            return self._pick_one_item(state)

    def _pick_one_item(self, state: SolverState) -> NDArray[np.int32]:
        """Return one item among the `top_k` highest contributions, after one pass over the dataset."""
        # both arrays below are ascending-index, so positions align
        contributions = state.not_selected_contribution_array
        if self._top_k == 1:
            return np.array([state.not_selected_index_array[np.argmax(contributions)]], dtype=np.int32)
        else:
            k_eff = min(self._top_k, len(contributions))
            top_positions = select_k_max(contributions, np.int32(k_eff))
            drawn = randint(n=np.int32(k_eff), k=np.int32(1), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
            return np.array([state.not_selected_index_array[top_positions[drawn[0]]]], dtype=np.int32)

    def _draw_round(
        self, state: SolverState, candidate_pool_size: int, k_remaining: int | np.int32
    ) -> NDArray[np.int32]:
        """Draw one round's batch for a non-empty selection: collect a candidate pool and draw from it.

        The batch holds at least 1 item and grows while the pool can still be shown to hold the
        dataset's best candidates. Every draw samples uniformly among the `top_k` best remaining pool
        candidates and advances the strategy's RNG state.
        """
        # --- candidate pool ---------------------
        # the candidate_pool_size highest-contribution not-selected items
        cand_idx, cand_val = state.top_not_selected_contributions(candidate_pool_size)
        # every item outside the pool is below the pool's lowest value, so a pool candidate still at
        # or above that value is among the dataset's best: it is the round's admission threshold
        threshold = np.float32(cand_val.min())
        if len(cand_idx) < candidate_pool_size:
            # the pool holds every remaining item, so nothing is outside it: draw until the pool is empty
            threshold = np.float32(-np.inf)

        # --- draw -------------------------------
        b_target = min(len(cand_idx), int(k_remaining))
        out_batch = np.empty(b_target, dtype=np.int32)
        top_positions = np.empty(self._top_k, dtype=np.int32)  # scratch for the draw loop's top-k positions
        n_drawn = _draw_from_pool(
            cand_idx,
            cand_val,
            np.int32(self._top_k),
            threshold,
            np.int64(b_target),
            state.distance_store,
            self._rng_state,
            out_batch,
            top_positions,
        )
        return out_batch[:n_drawn]

    @staticmethod
    def _are_rounds_supported(objective: DiversityObjective) -> bool:
        """Return whether rounds apply: every term of `objective` is a separation-family metric over one distance.

        The rule that ends a round relies on each contribution only falling as items are selected, and
        only that kind of objective guarantees it.
        """
        specs = objective.distinct_tracker_specs
        return len(specs) == 1 and specs[0].contribution_family == DiversityContributionFamily.SEPARATION


# =================================================================================================
#  Helpers
# =================================================================================================
# The drawing loop is numba-compiled, and numba cannot compile methods, so it lives here beside the class.
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
def _draw_from_pool(
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
        k_eff = min(np.int64(top_k), n_live)  # n_live >= 1: the loop stops at b_target <= count
        if k_eff < top_k and threshold > -np.inf:
            break  # too few live candidates to show the draw would range over the dataset's best
        if select_k_max_into(cand_val, n_live, k_eff, top_positions) < threshold:
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
