import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, randint
from max_div._core.metrics import DiversityContributionFamily, DiversityMetric
from max_div._core.metrics._distance import DISTANCE_STORE_TYPE, DistanceStore, get_distance
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


@numba.njit(
    numba.void(numba.int32[:], numba.float32[:], numba.int64, numba.int64),
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _move_top_to_front(
    cand_idx: NDArray[np.int32], cand_val: NDArray[np.float32], n_top: np.int64, n_remaining: np.int64
) -> None:
    """Reorder the first `n_top` pool slots to hold the highest values among the first `n_remaining`, descending."""
    for s in range(n_top):
        best = s
        for c in range(s + 1, n_remaining):
            if cand_val[c] > cand_val[best]:
                best = c
        if best != s:
            cand_val[s], cand_val[best] = cand_val[best], cand_val[s]
            cand_idx[s], cand_idx[best] = cand_idx[best], cand_idx[s]


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
) -> np.int64:
    """Draw items from the candidate pool into `out_batch` until the pool goes stale; return the count drawn.

    `threshold` is the pool's admission value — the lowest contribution it held when the round
    opened. Every item outside the pool was below it then, and contributions only fall as items are
    selected, so any pool candidate still at or above it is among the whole dataset's best. The
    round therefore draws while the pool holds a `top_k`-th best live candidate at or above
    `threshold`, which makes each draw range over the same candidates the per-pick construction
    would offer, and ends as soon as that stops being provable — including when the pool has been
    drawn down below `top_k` live candidates, unless it holds every remaining item anyway.

    After each draw the remaining pool is refreshed against the drawn item, so a candidate that the
    draw brought close to the selection drops out of contention immediately.
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
        _move_top_to_front(cand_idx, cand_val, k_eff, n_live)
        if cand_val[k_eff - 1] < threshold:
            break
        # a single candidate needs no draw, matching `InitFarthestPoint`'s argmax path
        pick_pos = 0 if k_eff == 1 else randint(np.int32(k_eff), np.int32(1), False, p_uniform, rng_state)[0]
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
    pass over the dataset, then draws from that pool without touching the dataset again: each draw
    samples uniformly among the pool's `top_k` best remaining candidates and refreshes the rest
    against the drawn item. The round ends once the pool can no longer be shown to hold the
    dataset's best candidates, and the solver applies the whole batch in one tracker update.

    Every draw ranges over the same candidates `InitFarthestPoint` would offer at that point (see
    `_draw_round` for why), so the two strategies build selections of the same quality; they differ
    in which of the equally-good candidates a given seed happens to draw. The gain is that a pass
    over the dataset serves a whole batch instead of a single pick, which at large n makes this
    several times faster.

    The construction is tailored to separation-family diversity metrics, whose contributions only
    fall as items are selected — the property the round's stopping rule rests on; other metric
    families are rejected when the solver is built. Constraints are ignored by design, like
    `InitFarthestPoint`.

    Parameters:
    - top_k (int): every draw samples uniformly among the `top_k` best remaining candidates — the
                   meaning it has on `InitFarthestPoint`; 1 keeps the exact greedy pick. This is
                   the knob that decorrelates runs sharing a seed source. (default: 8)
    - batch_size (int): how many candidates a round collects, which bounds how many items it can
                        draw. Deeper pools reach further down the contribution ranking, so rounds
                        run longer and fewer passes over the dataset are needed, at the cost of
                        refreshing more candidates per draw. It cannot affect the selection's
                        quality, only the time spent. (default: 1024)
    """

    def __init__(self, top_k: int = 8, batch_size: int = 1024) -> None:
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
        n_drawn = _draw_round(
            cand_idx,
            cand_val,
            np.int32(self._top_k),
            threshold,
            np.int64(b_target),
            state.distance_store,
            self._rng_state,
            out_batch,
        )
        return out_batch[:n_drawn]
