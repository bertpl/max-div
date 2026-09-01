import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, randint
from max_div._core.metrics import DiversityContributionFamily, DiversityMetric
from max_div._core.metrics._distance import DISTANCE_STORE_TYPE, DistanceStore, get_distance
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


@numba.njit(
    numba.void(numba.float32[:], numba.int32[:], numba.int64, numba.int64),
    inline="always",
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _sift_down(values: NDArray[np.float32], positions: NDArray[np.int32], size: np.int64, start: np.int64) -> None:
    """Restore the min-heap property below `start` in a heap of positions ordered by `values`."""
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
    """Fill `out_positions` with the `n_top` highest values' positions among the first `n_live`; return the lowest.

    The positions come back in unspecified order. Ties keep the earlier position, so with
    `n_top == 1` the result is the first maximum. The caller owns `out_positions` (at least
    `n_top` long), so a draw allocates nothing.
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
    draw range over the candidates the per-pick construction would offer. The round ends when the
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
        # a single candidate needs no draw, matching `InitFarthestPoint`'s argmax path
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


class InitFarthestPointBatched(InitializationStrategy):
    """Initialize by farthest-point sampling in rounds, drawing many items per pass over the dataset.

    A round collects the `batch_size` highest-contribution not-selected items into a pool with one
    pass over the dataset, then draws from that pool without touching the dataset again, ending
    once the pool can no longer be shown to hold the dataset's best candidates (see `_draw_round`).
    The solver applies the whole batch in one tracker update. Every draw ranges over the same
    candidates `InitFarthestPoint` would offer, so selections are of the same quality but not the
    same, and the batched strategy is several times faster at large n.

    Separation-family diversity metrics only — their contributions only fall as items are selected,
    which the stopping rule rests on; other families are rejected when the solver is built.
    Constraints are ignored by design, like `InitFarthestPoint`.

    Parameters:
    - top_k (int): every draw samples uniformly among the `top_k` best remaining candidates — the
                   meaning it has on `InitFarthestPoint`; 1 keeps the exact greedy pick. (default: 8)
    - batch_size (int): how many candidates a round collects, which bounds how many items it can
                        draw. A deeper pool needs fewer passes over the dataset but refreshes more
                        candidates per draw; above a few hundred the extra refreshing costs more
                        than the saved passes. `batch_size` cannot affect the selection's quality,
                        only the time spent. (default: 256)
    """

    def __init__(self, top_k: int = 8, batch_size: int = 256) -> None:
        """Create the strategy.

        Raises:
            ValueError: If `top_k` is below 1, or `batch_size` is below `top_k` (a round could then
                not offer a full draw).
        """
        super().__init__()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if batch_size < top_k:
            raise ValueError(f"batch_size must be >= top_k ({top_k}), got {batch_size}")
        self._top_k = top_k
        self._batch_size = batch_size

    def validate_diversity_metric(self, diversity_metric: DiversityMetric) -> None:
        """Reject metric families the round heuristics are not tailored to."""
        if diversity_metric.contribution_family != DiversityContributionFamily.SEPARATION:
            raise ValueError(
                f"InitFarthestPointBatched does not support diversity metric {diversity_metric}: "
                "the heuristics of the algorithm are tailored to separation-based diversity metrics. "
                "Use InitializationStrategy.farthest_point() instead."
            )

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        if state.n_selected == 0:
            return randint(n=state.n, k=np.int32(1), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
        cand_idx, cand_val = state.top_not_selected_contributions(self._batch_size)
        # a pool holding every remaining item has nothing outside it to be measured against
        threshold = np.float32(-np.inf) if len(cand_idx) < self._batch_size else np.float32(cand_val.min())
        b_target = min(len(cand_idx), int(k_remaining))
        out_batch = np.empty(b_target, dtype=np.int32)
        top_positions = np.empty(self._top_k, dtype=np.int32)
        n_drawn = _draw_round(
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
